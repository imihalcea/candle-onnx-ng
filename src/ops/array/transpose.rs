use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};

pub(crate) struct Transpose;
impl OnnxOp for Transpose {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let input = node.get_input(0)?;
        let output = match node.get_attr_opt::<[i64]>("perm")? {
            None => input.t()?,
            Some(perm) => {
                let perm = perm.iter().map(|&v| v as usize).collect::<Vec<_>>();
                input.permute(perm)?
            }
        };

        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}
