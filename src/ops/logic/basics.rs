use crate::ops::compute_node::ComputeNode;
use crate::ops::{OnnxOp, OnnxOpError, OpOutput};

pub(crate) struct Xor;
impl OnnxOp for Xor {
    // https://onnx.ai/onnx/operators/onnx__Xor.html
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let a = node.get_input(0)?.gt(0_u8)?;
        let b = node.get_input(1)?.gt(0_u8)?;
        let out = a.broadcast_add(&b)?.eq(1_u8)?;

        let output_name = node.get_output(0)?;
        Ok(OpOutput::Single(output_name.clone(), out))
    }
}

pub(crate) struct Not;

impl OnnxOp for Not {
    // https://onnx.ai/onnx/operators/onnx__Not.html
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let xs = node.get_input(0)?;
        let out = xs.eq(&xs.zeros_like()?)?;
        let output_name = node.get_output(0)?;
        Ok(OpOutput::Single(output_name.clone(), out))
    }
}
