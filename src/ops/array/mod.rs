use crate::ops::{OnnxOpError, OnnxOpRegistry};

mod shape;
mod transpose;

pub(crate) fn register(registry: &mut OnnxOpRegistry) -> Result<(), OnnxOpError> {
    registry.insert("Reshape", Box::new(shape::Reshape))?;
    registry.insert("Transpose", Box::new(transpose::Transpose))?;
    Ok(())
}
