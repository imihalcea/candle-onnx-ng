use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};
pub(crate) struct Sqrt;
impl OnnxOp for Sqrt {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sqrt
        let xs = node.get_input(0)?;
        let output = xs.sqrt()?;
        let output_name = node.get_output(0)?;
        Ok(OpOutput::Single(output_name.clone(), output))
    }
}
