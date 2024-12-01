use crate::ops::{OnnxOpError, OnnxOpRegistry};
mod softmax;
mod dropout;
mod maxpool;

pub(crate) fn register(registry: &mut OnnxOpRegistry) -> Result<(), OnnxOpError> {
    registry.insert("LogSoftmax", Box::new(softmax::LogSoftmax))?;
    registry.insert("Softmax", Box::new(softmax::Softmax))?;
    registry.insert("Dropout", Box::new(dropout::Dropout))?; //training not supported
    Ok(())
}
