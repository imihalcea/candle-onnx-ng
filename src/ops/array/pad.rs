use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};
use candle_core::Tensor;

pub(crate) struct Pad;

impl OnnxOp for Pad {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let mode = node.get_attr_opt("mode")?.unwrap_or("constant");
        let data = node.get_input(0)?;
        let pads = node.get_input(1)?;
        if node.input_len() > 2 {
            let err_msg = format!(
                "unsupported number of inputs {} for Pad node {:?}, expected 2",
                node.input_len(),
                node.name
            );
            return Err(OnnxOpError::InvalidInput(err_msg));
        }
        if pads.rank() != 1 {
            let err_msg = format!("Pad expects 'pads' input to be 1D vector: {pads:?}");
            return Err(OnnxOpError::InvalidInput(err_msg));
        }
        if pads.dim(0)? != 2 * data.rank() {
            let err_msg = format!("Pad expects 'pads' input len to be 2 * rank of 'data' input: pads: {}, data rank: {}", pads, data.rank());
            return Err(OnnxOpError::InvalidInput(err_msg));
        }

        let pads = pads.to_vec1::<i64>()?;
        let (pads_pre, pads_post) = pads.split_at(pads.len() / 2);

        match mode {
            "reflect" => {
                let mut out = data.clone();
                for (i, &dim) in data.dims().iter().enumerate().rev() {
                    if pads_pre[i] == 0 && pads_post[i] == 0 {
                        continue;
                    }
                    fn zigzag(min: i64, max: i64) -> impl Iterator<Item = i64> {
                        std::iter::repeat((min..max).chain((min + 1..=max).rev())).flatten()
                    }
                    let idx = if dim > 1 {
                        let cycle_len = dim * 2 - 2;
                        let skip = cycle_len - ((pads_pre[i] as usize) % cycle_len);
                        let idx = zigzag(0, (dim - 1) as i64)
                            .skip(skip)
                            .take((pads_pre[i] as usize) + dim + (pads_post[i] as usize));
                        Tensor::from_iter(idx, out.device())?
                    } else {
                        Tensor::full(0i64, (dim,), out.device())?
                    };

                    out = out.index_select(&idx, i)?;
                }

                let output_name = node.get_output(0)?;
                Ok(OpOutput::Single(output_name.clone(), out))
            }
            _ => {
                let err_msg = format!(
                    "unsupported 'mode' value {mode:?} for Pad node {:?}",
                    node.name
                );
                Err(OnnxOpError::UnsupportedAttribute(err_msg))
            }
        }
    }
}
