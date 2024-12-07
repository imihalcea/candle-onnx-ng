use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};
use candle_core::Tensor;
pub(crate) struct Shape;
impl OnnxOp for Shape {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Shape
        let xs = node.get_input(0)?;
        let start = node.get_attr_opt::<i64>("start")?.copied().unwrap_or(0);
        let end = node.get_attr_opt::<i64>("end")?.copied().unwrap_or(-1);
        let start = xs.normalize_axis(start)?;
        let end = xs.normalize_axis(end)?;
        let mut dims = vec![];
        for idx in start..=end {
            dims.push(xs.dim(idx)? as i64)
        }
        let dims = Tensor::from_vec(dims, xs.rank(), xs.device())?;
        let output_name = node.get_output(0)?;
        Ok((output_name.clone(), dims))
    }
}
