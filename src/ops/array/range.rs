use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};
use candle_core::{DType, Device, Tensor};
pub(crate) struct Range;
impl OnnxOp for Range {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Range
        let start = node.get_input(0)?;
        let limit = node.get_input(1)?;
        let delta = node.get_input(2)?;

        macro_rules! arange_step {
            ($t: ty) => {
                Tensor::arange_step(
                    start.to_vec0::<$t>()?,
                    limit.to_vec0::<$t>()?,
                    delta.to_vec0::<$t>()?,
                    &Device::Cpu, //TO DO ???
                )?
            };
        }

        let output = match start.dtype() {
            DType::U8 => arange_step!(u8),
            DType::U32 => arange_step!(u32),
            DType::I64 => arange_step!(i64),
            DType::BF16 => arange_step!(f32),
            DType::F16 => arange_step!(f32),
            DType::F32 => arange_step!(f32),
            DType::F64 => arange_step!(f64),
        };

        let output_name = node.get_output(0)?;
        Ok((output_name.clone(), output))
    }
}
