use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};
use candle_core::{Device, Tensor};

pub(crate) struct Gemm;
impl OnnxOp for Gemm {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm
        let a = node.get_input(0)?;
        let b = node.get_input(1)?;
        let c = node.get_input(2)?;
        //FIXME: according to the spec c is optional, but we are not handling it

        let alpha = node.get_attr_opt::<f32>("alpha")?.copied().unwrap_or(1.0);
        let beta = node.get_attr_opt::<f32>("beta")?.copied().unwrap_or(1.0);

        let alpha = Tensor::full(alpha, a.shape(), &Device::Cpu)?;
        let beta = Tensor::full(beta, c.shape(), &Device::Cpu)?;

        let trans_a = node.get_attr_opt::<i64>("transA")?.copied().unwrap_or(0);
        let trans_b = node.get_attr_opt::<i64>("transB")?.copied().unwrap_or(0);

        let a = if trans_a == 0 { a.clone() } else { a.t()? };
        let b = if trans_b == 0 { b.clone() } else { b.t()? };

        let output = a
            .broadcast_mul(&alpha)?
            .broadcast_matmul(&b)?
            .broadcast_add(&c.broadcast_mul(&beta)?)?;

        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}
