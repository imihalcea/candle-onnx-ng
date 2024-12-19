use candle_core::DType;
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
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gelu
        let input = node.get_input(0)?;
        let output = input.gelu_erf()?;
        let output_name = node.get_output(0)?;
        Ok((output_name.clone(), output))
    }
}

pub(crate) struct Relu;

impl OnnxOp for Relu {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu
        let input = node.get_input(0)?;
        let output = input.relu()?;
        let output_name = node.get_output(0)?;
        Ok((output_name.clone(), output))
    }
}

pub(crate) struct LeakyRelu;

impl OnnxOp for LeakyRelu {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let input = &node.get_input(0)?;
        let dt = input.dtype();
        match dt {
            DType::U8 | DType::U32 | DType::I64 => {
                let err_msg = format!(
                            "unsupported dtype {}, only float types are allowed for LeakyRelu",
                            dt.as_str()
                        );
                return Err(OnnxOpError::InvalidInput(err_msg));
            }
            DType::BF16 | DType::F16 | DType::F32 | DType::F64 => {}
        }
        let alpha = node.get_attr_opt::<f32>("alpha")?
            .copied()
            .unwrap_or(0.01);

        let output = candle_nn::ops::leaky_relu(input, alpha.into())?;
        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}