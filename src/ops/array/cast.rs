use crate::onnx::tensor_proto::DataType;
use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};
use crate::parser;
use candle_core::DType;
pub(crate) struct Cast;
impl OnnxOp for Cast {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast
        let input = node.get_input(0)?;
        let dt = *node.get_attr::<i64>("to")?;
        let dtype = match DataType::try_from(dt as i32) {
            Ok(DataType::Int32) => DType::I64,
            Ok(dt) => match parser::dtype(dt) {
                Some(dt) => dt,
                None => {
                    let err_msg = format!("unsupported 'to' value {dt:?} for cast {}", node.name);
                    return Err(OnnxOpError::ComputationFailed(err_msg));
                }
            },
            Err(_) => {
                let err_msg = format!("unsupported 'to' value {dt:?} for cast {}", node.name);
                return Err(OnnxOpError::ComputationFailed(err_msg));
            }
        };
        let output = input.to_dtype(dtype)?;
        let output_name = node.get_output(0)?;
        Ok((output_name.clone(), output))
    }
}
