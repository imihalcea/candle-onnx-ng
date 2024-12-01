use crate::ops::{OnnxOpError, OnnxOpRegistry};

mod shape;

pub(crate) fn register(registry: &mut OnnxOpRegistry) -> Result<(), OnnxOpError>  {
    registry.insert("Reshape", Box::new(shape::Reshape))?;
    Ok(())
}