use crate::ops::compute_node::ComputeNode;
use crate::ops::OnnxOpError::ComputationFailed;
use crate::ops::{OnnxOp, OnnxOpError, OpOutput};

pub(crate) struct Xor;
impl OnnxOp for Xor {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let a = node.get_input(0)?.gt(0_u8)?;
        let b = node.get_input(0)?.gt(0_u8)?;
        let out = a.broadcast_add(&b)?.eq(1_u8)?;

        let output_name = node.get_output(0)?;
        Ok((output_name.clone(), out))
    }
}
