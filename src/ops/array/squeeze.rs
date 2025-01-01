use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};
use candle_core::Error;

pub(crate) struct Squeeze;

impl OnnxOp for Squeeze {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let xs = node.get_input(0)?;
        let mut axes = if node.input_len() <= 1 {
            // contract all the dimensions with size 1 except the batch dim.
            xs.dims()
                .iter()
                .enumerate()
                .flat_map(|(idx, &s)| if s == 1 && idx > 0 { Some(idx) } else { None })
                .collect()
        } else {
            node.get_input(1)?
                .to_vec1::<i64>()?
                .iter()
                .map(|&i| xs.normalize_axis(i))
                .collect::<Result<Vec<_>, Error>>()?
        };
        axes.sort();

        let mut xs = xs.clone();
        for &axis in axes.iter().rev() {
            xs = xs.squeeze(axis)?
        }

        let output_name = node.get_output(0)?;
        Ok(OpOutput::Single(output_name.clone(), xs))
    }
}
