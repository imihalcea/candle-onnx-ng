use crate::ops::{OnnxOpError, OnnxOpRegistry};

mod sign;
mod add;
mod sub;

pub(crate) fn register(registry: &mut OnnxOpRegistry) -> Result<(), OnnxOpError>  {
    registry.insert("Sign", Box::new(sign::Sign))?;
    registry.insert("Add", Box::new(add::Add))?;
    registry.insert("Sub", Box::new(sub::Sub))?;
    Ok(())
}