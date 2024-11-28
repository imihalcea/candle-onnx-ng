use crate::ops::compute_node::ComputeNode;
use crate::ops::{OnnxOp, OnnxOpError, OpOutput};

pub(crate) struct Exp;
impl OnnxOp for Exp {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let xs = node.get_input(0)?;
        let output = xs.exp()?;
        
        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}