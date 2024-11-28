use crate::ops::{OnnxOpError, OnnxOpRegistry};

mod basics;

pub(crate) fn register(registry: &mut OnnxOpRegistry) -> Result<(), OnnxOpError>  {
    registry.insert("Xor", Box::new(basics::Xor))?;
    Ok(())
}