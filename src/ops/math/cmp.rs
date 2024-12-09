use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};

pub(crate) struct Equal;
impl OnnxOp for Equal {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let input0 = node.get_input(0)?;
        let input1 = node.get_input(1)?;
        let output = input0.broadcast_eq(input1)?;

        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}

pub(crate) struct Greater;
impl OnnxOp for Greater {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Greater
        let a = node.get_input(0)?;
        let b = node.get_input(1)?;

        let output = a.broadcast_gt(b)?;

        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}

pub(crate) struct Less;
impl OnnxOp for Less {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Less
        let a = node.get_input(0)?;
        let b = node.get_input(1)?;

        let output = a.broadcast_lt(b)?;

        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}
