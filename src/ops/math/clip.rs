use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};

pub(crate) struct Clip;
impl OnnxOp for Clip {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let xs = node.get_input(0)?;

        let xs = if let Some(mins) = node.get_opt(1) {
            xs.broadcast_maximum(mins)?
        } else {
            xs.clone()
        };
        let xs = if let Some(maxs) = node.get_opt(2) {
            xs.broadcast_minimum(maxs)?
        } else {
            xs.clone()
        };

        let output_name = node.get_output(0)?;
        Ok((output_name.clone(), xs))
    }
}
