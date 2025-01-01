use crate::onnx::attribute_proto::AttributeType;
use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};
use crate::parser;

pub(crate) struct Constant;

impl OnnxOp for Constant {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let value = match node.get_attr_definition("value") {
            None => {
                // TODO: support sparse_value etc.
                return Err(OnnxOpError::InvalidAttribute("value".to_string()));
            }
            Some(value) => value,
        };
        let output = match value.r#type() {
            AttributeType::Tensor => {
                let t = value.t.as_ref().unwrap();
                parser::get_tensor(t, &node.name)?
            }
            rtype => {
                return Err(OnnxOpError::UnsupportedAttribute(format!(
                    "unsupported 'value' type {rtype:?} for {}",
                    node.name
                )))
            }
        };

        let output_name = node.get_output(0)?;
        Ok(OpOutput::Single(output_name.clone(), output))
    }
}
