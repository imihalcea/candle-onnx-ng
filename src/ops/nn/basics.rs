use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};

pub(crate) struct Sigmoid;

impl OnnxOp for Sigmoid {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sigmoid
        let input = node.get_input(0)?;
        let output = candle_nn::ops::sigmoid(input)?;
        let output_name = node.get_output(0)?;
        Ok((output_name.clone(), output))
    }
}

pub(crate) struct Gelu;

impl OnnxOp for Gelu {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sigmoid
        let input = node.get_input(0)?;
        let output = input.gelu_erf()?;
        let output_name = node.get_output(0)?;
        Ok((output_name.clone(), output))
    }
}
