use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};
use candle_core::Tensor;

pub(crate) struct Gather;

impl OnnxOp for Gather {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather
        let xs = node.get_input(0)?;
        let indices = node.get_input(1)?;
        let axis = node.get_attr_opt::<i64>("axis")?.copied().unwrap_or(0);
        let axis = xs.normalize_axis(axis)?;

        // index_select does not support negative indices, so normalize them
        // to positive indices.
        let indices = &{
            let zeros = Tensor::zeros(indices.shape(), indices.dtype(), indices.device())?;
            let max =
                Tensor::new(xs.dims()[axis] as i64, indices.device())?.to_dtype(indices.dtype())?;
            let mask = indices.lt(&zeros)?;
            mask.to_dtype(indices.dtype())?
                .broadcast_mul(&max)?
                .add(indices)?
        };

        // In Pytorch or Numpy this can be done by indexing the xs tensor using the indices
        // tensor directly, but candle does not support tensor indexing at the moment, so
        // some workarounds must be done.
        let xs = match indices.dims() {
            [] => {
                let index = indices.to_vec0::<i64>()? as usize;
                xs.narrow(axis, index, 1)?.squeeze(axis)?
            }
            [_] => xs.index_select(indices, axis)?,
            [first, _] => {
                let mut v = Vec::with_capacity(*first);
                for i in 0..*first {
                    v.push(xs.index_select(&indices.get(i)?, axis)?)
                }
                Tensor::stack(&v, axis)?
            }
            _ => {
                // TODO: Provide an op to handle the ONNX generalized gather op ideally in a
                // differentiable way.
                todo!("implement gather for {xs:?} {indices:?} axis {axis}")
            }
        };

        let output_name = node.get_output(0)?;
        Ok((output_name.clone(), xs))
    }
}

pub(crate) struct GatherElements;

impl OnnxOp for GatherElements {

    // https://onnx.ai/onnx/operators/onnx__GatherElements.html#gatherelements
    // A Note to fellow lurkers:
    // The numpy based `gather_elements` implementation in `onnx` tests [here](https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/gatherelements.py)
    // and examples is incorrect.
    // Use `torch.gather` for the validating/ verifying against the proper behaviour

    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let data = node.get_input(0)?;
        let indices = node.get_input(1)?;

        let rank = data.rank();
        if rank != indices.rank() {
            return Err(OnnxOpError::InvalidInput(format!(
                "indices must have same rank as input data. Data rank [{}] != indices rank [{}]",
                data.rank(),
                indices.rank()
            )));
        }

        let axis = {
            let axis_i64 = node.get_attr_opt::<i64>("axis")?
                .copied()
                .unwrap_or(0);
            let axis = data.normalize_axis(axis_i64)?;

            if axis >= rank {
                return Err(OnnxOpError::InvalidInput(format!(
                    "axis ({}) out of accepted range [-rank, rank-1] which was [-{rank}, {}]",
                    axis_i64,
                    rank - 1
                )));
            }
            axis
        };

        // index_select does not support negative indices, so normalize them
        // to positive indices.
        let indices = &{
            let zeros = Tensor::zeros(indices.shape(), indices.dtype(), indices.device())?;
            let max = Tensor::new(data.dims()[axis] as i64, indices.device())?
                .to_dtype(indices.dtype())?;
            let mask = indices.lt(&zeros)?;
            mask.to_dtype(indices.dtype())?
                .broadcast_mul(&max)?
                .add(indices)?
        };

        let output = data.gather(indices, axis)?;
        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}
