use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};
use candle_core::Error;
pub(crate) struct Reshape;
impl OnnxOp for Reshape {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let input0 = node.get_input(0)?;
        let input1 = node.get_input(1)?.to_vec1::<i64>()?;
        // TODO: Check that there is at most a single -1 or 0, handle other neg values.
        let mut other_than_minus1 = 1usize;
        for &v in input1.iter() {
            if v != -1 && v != 0 {
                other_than_minus1 *= v as usize
            }
        }

        let input1 = input1
            .iter()
            .enumerate()
            .map(|(idx, &v)| match v {
                -1 => Ok(input0.elem_count() / other_than_minus1),
                0 => input0.dim(idx),
                _ => Ok(v as usize),
            })
            .collect::<Result<Vec<usize>, Error>>()?;

        let output = input0.reshape(input1)?;

        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}
