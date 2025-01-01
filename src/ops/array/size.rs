use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};
use candle_core::Tensor;
pub(crate) struct Size;
impl OnnxOp for Size {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Size
        let data = node.get_input(0)?;
        let size: usize = data.dims().iter().product();
        let output = Tensor::from_slice(&[size as i64], (), data.device())?;
        let output_name = node.get_output(0)?;
        Ok(OpOutput::Single(output_name.clone(), output))
    }
}
