use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};

pub(crate) struct Flatten;
impl OnnxOp for Flatten {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        //  https://github.com/onnx/onnx/blob/main/docs/Operators.md#flatten
        let input = node.get_input(0)?;
        let axis = node.get_attr_opt::<i64>("axis")?.copied().unwrap_or(1) as usize;

        let first_part: usize = input.shape().dims().iter().take(axis).product();
        let end_index = input.shape().dims().iter().product::<usize>();
        let new_shape = (first_part, end_index / first_part);

        let output = input.reshape(new_shape)?;
        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}
