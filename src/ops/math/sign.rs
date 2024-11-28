use crate::ops::compute_node::ComputeNode;
use crate::ops::{OnnxOp, OnnxOpError, OpOutput};

pub(crate) struct Sign;
impl OnnxOp for Sign {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let input = node.get_input(0)?;
        let output = input.sign()?;
        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}
