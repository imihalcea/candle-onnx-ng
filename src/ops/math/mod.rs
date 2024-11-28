use crate::ops::{OnnxOpError, OnnxOpRegistry};
mod basics;

pub(crate) fn register(registry: &mut OnnxOpRegistry) -> Result<(), OnnxOpError>  {
    registry.insert("Add", Box::new(basics::Add))?;
    registry.insert("Sub", Box::new(basics::Sub))?;
    registry.insert("Mul", Box::new(basics::Mul))?;
    registry.insert("Div", Box::new(basics::Div))?;
    registry.insert("Exp", Box::new(basics::Exp))?;
    registry.insert("Pow", Box::new(basics::Pow))?;
    registry.insert("Sign", Box::new(basics::Sign))?;
    Ok(())
}