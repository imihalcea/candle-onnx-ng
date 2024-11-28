use crate::ops::{OnnxOpError, OnnxOpRegistry};

mod sign;
mod add;

pub(crate) fn register(registry: &mut OnnxOpRegistry) -> Result<(), OnnxOpError>  {
    registry.insert("Sign", Box::new(sign::Sign))?;
    registry.insert("Add", Box::new(add::Add))?;
    Ok(())
}