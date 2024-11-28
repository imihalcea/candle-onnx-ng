use crate::ops::{OnnxOpError, OnnxOpRegistry};

mod basics;

pub(crate) fn register(registry: &mut OnnxOpRegistry) -> Result<(), OnnxOpError>  {
    registry.insert("Xor", Box::new(basics::Xor))?;
    registry.insert("Not", Box::new(basics::Not))?;
    Ok(())
}