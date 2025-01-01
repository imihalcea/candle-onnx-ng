use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};
use candle_core::DType;

pub(crate) struct ArgMin;

impl OnnxOp for ArgMin {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let input = node.get_input(0)?;
        let axis_i64: i64 = node.get_attr_opt("axis")?.copied().unwrap_or(0);
        let rank_i64: i64 = input.rank().try_into().unwrap();
        if axis_i64 < -rank_i64 || axis_i64 >= rank_i64 {
            let err_msg = format!(
                "axis ({}) out of accepted range [-rank, rank-1] which was [{}, {}]",
                axis_i64,
                -rank_i64,
                rank_i64 - 1
            );
            return Err(OnnxOpError::InvalidAttribute(err_msg));
        }
        let axis = input.normalize_axis(axis_i64)?;
        let keepdims: i64 = node.get_attr_opt("keepdims")?.copied().unwrap_or(1);
        let select_last_index: i64 = node
            .get_attr_opt("select_last_index")?
            .copied()
            .unwrap_or(0);
        if select_last_index == 1 {
            let err_msg = "select_last_index for ArgMin is currently not supported".to_string();
            return Err(OnnxOpError::InvalidAttribute(err_msg));
        }
        let output = match keepdims {
            1 => input.argmin_keepdim(axis)?.to_dtype(DType::I64)?,
            _ => input.argmin(axis)?.to_dtype(DType::I64)?,
        };

        let output_name = node.get_output(0)?.clone();

        Ok(OpOutput::Single(output_name, output))
    }
}

pub(crate) struct ArgMax;

impl OnnxOp for ArgMax {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let input = node.get_input(0)?;
        let axis_i64: i64 = node.get_attr_opt("axis")?.copied().unwrap_or(0);
        let rank_i64: i64 = input.rank().try_into().unwrap();
        if axis_i64 < -rank_i64 || axis_i64 >= rank_i64 {
            let err_msg = format!(
                "axis ({}) out of accepted range [-rank, rank-1] which was [{}, {}]",
                axis_i64,
                -rank_i64,
                rank_i64 - 1
            );
            return Err(OnnxOpError::InvalidAttribute(err_msg));
        }
        let axis = input.normalize_axis(axis_i64)?;
        let keepdims: i64 = node.get_attr_opt("keepdims")?.copied().unwrap_or(1);
        let select_last_index: i64 = node
            .get_attr_opt("select_last_index")?
            .copied()
            .unwrap_or(0);
        if select_last_index == 1 {
            let err_msg = "select_last_index for ArgMin is currently not supported".to_string();
            return Err(OnnxOpError::UnsupportedAttribute(err_msg));
        }
        let output = if keepdims == 1 {
            input.argmax_keepdim(axis)?
        } else {
            input.argmax(axis)?
        }
        .to_dtype(DType::I64)?;

        let output_name = node.get_output(0)?.clone();
        Ok(OpOutput::Single(output_name, output))
    }
}
