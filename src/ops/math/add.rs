use crate::ops::compute_node::ComputeNode;
use crate::ops::{OnnxOp, OnnxOpError, OpOutput};

pub(crate) struct Add;
impl OnnxOp for Add {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let input0 = &node.get_input(0)?;
        let input1 = &node.get_input(1)?;
        let output = input0.broadcast_add(input1)?;

        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}