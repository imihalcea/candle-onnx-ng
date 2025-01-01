use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};
use candle_core::Tensor;

pub(crate) struct Concat;
impl OnnxOp for Concat {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Concat
        let inputs = node.get_all_inputs()?;
        let axis: i64 = *node.get_attr("axis")?;
        if inputs.is_empty() {
            return Err(OnnxOpError::InvalidInput("inputs are empty".to_string()));
        };
        let axis = inputs[0].normalize_axis(axis)?;
        let output = Tensor::cat(&inputs, axis)?;
        let output_name = node.get_output(0)?;
        Ok(OpOutput::Single(output_name.clone(), output))
    }
}
