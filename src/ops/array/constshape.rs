use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};
use candle_core::{DType, Device, Tensor};
pub(crate) struct ConstantOfShape;
impl OnnxOp for ConstantOfShape {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let input = node.get_input(0)?;
        let value = node
            .get_attr_opt_owned::<Tensor>("value")?
            .unwrap_or(Tensor::zeros((), DType::F32, &Device::Cpu)?);

        let xs =
            Tensor::ones(input.shape(), value.dtype(), input.device())?.broadcast_mul(&value)?;

        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), xs))
    }
}
