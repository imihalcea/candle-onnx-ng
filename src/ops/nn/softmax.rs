use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};
pub(crate) struct LogSoftmax;
impl OnnxOp for LogSoftmax {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let input = node.get_input(0)?;
        let output = match node.get_attr_opt::<i64>("axis")? {
            None => candle_nn::ops::softmax_last_dim(input)?,
            Some(&axis) => {
                let axis = input.normalize_axis(axis)?;
                candle_nn::ops::log_softmax(input, axis)?
            }
        };

        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}

pub(crate) struct Softmax;
impl OnnxOp for Softmax {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let input = node.get_input(0)?;
        let output =  match node.get_attr_opt::<i64>("axis")? {
            None => candle_nn::ops::softmax_last_dim(input)?,
            Some(&axis) => {
                let axis = input.normalize_axis(axis)?;
                candle_nn::ops::softmax(input, axis)?
            }
        };

        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}
