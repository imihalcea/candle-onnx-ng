use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};

pub(crate) struct Dropout;
impl OnnxOp for Dropout {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let input = node.get_input(0)?;
        let output = input.clone();
        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}