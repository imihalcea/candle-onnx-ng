use crate::ops::compute_node::ComputeNode;
use crate::ops::{OnnxOp, OnnxOpError, OpOutput};
use candle_core::DType;

pub(crate) struct CumSum;
impl OnnxOp for CumSum {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        //https://github.com/onnx/onnx/blob/main/docs/Operators.md#CumSum
        //TODO: Support exclusive and reverse
        let exclusive = node.get_attr_opt::<i64>("exclusive")?.copied().unwrap_or(0);
        let reverse = node.get_attr_opt::<i64>("reverse")?.copied().unwrap_or(0);
        if exclusive != 0 {
            let err_msg = "only exclusive == 0 is supported in CumSum";
            return Err(OnnxOpError::InvalidInput(err_msg.to_string()));
        }
        if reverse != 0 {
            let err_msg = "only reverse == 0 is supported in CumSum";
            return Err(OnnxOpError::InvalidInput(err_msg.to_string()));
        }
        let input = node.get_input(0)?;
        let axis = node.get_input(1)?.to_dtype(DType::U32)?.to_vec0::<u32>()?;
        let output = input.cumsum(axis as usize)?;
        let output_name = node.get_output(0)?;
        Ok((output_name.clone(), output))
    }
}
