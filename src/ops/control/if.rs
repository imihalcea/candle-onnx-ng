use crate::onnx::GraphProto;
use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};

pub(crate) struct If;

impl OnnxOp for If {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        // protobuf encodes boolean false as 0 and true as 1
        let cond = node.get_input(0)?.get(0)?.to_scalar::<u8>()?;
        let attr_name = if cond != 0 {
            "then_branch"
        } else {
            "else_branch"
        };
        let sub_graph = node.get_attr::<GraphProto>(attr_name)?;
        if sub_graph.output.len() != node.output_len() {
            let err_msg = format!(
                "If node {:?} is malformed: branch outputs ({}) don't match node outputs ({})",
                node.name,
                sub_graph.output.len(),
                node.output_len()
            );
            return Err(OnnxOpError::MalformedOp(err_msg));
        }
        Ok(OpOutput::Branch(attr_name.to_string()))
    }
}
