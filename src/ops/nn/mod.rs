use crate::ops::{OnnxOpError, OnnxOpRegistry};
mod log_softmax;

pub(crate) fn register(registry: &mut OnnxOpRegistry) -> Result<(), OnnxOpError> {
    registry.insert("LogSoftmax", Box::new(log_softmax::LogSoftmax))?;
    Ok(())
}
