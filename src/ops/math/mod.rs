use crate::ops::{OnnxOpError, OnnxOpRegistry};

mod sign;
mod add;
mod sub;
mod mul;

pub(crate) fn register(registry: &mut OnnxOpRegistry) -> Result<(), OnnxOpError>  {
    registry.insert("Sign", Box::new(sign::Sign))?;
    registry.insert("Add", Box::new(add::Add))?;
    registry.insert("Sub", Box::new(sub::Sub))?;
    registry.insert("Mul", Box::new(mul::Mul))?;
    Ok(())
}