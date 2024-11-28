use crate::ops::{OnnxOpError, OnnxOpRegistry};

mod sign;

pub(crate) fn register(registry: &mut OnnxOpRegistry) -> Result<(), OnnxOpError>  {
    registry.insert("Sign", Box::new(sign::Sign))?;
    Ok(())
}