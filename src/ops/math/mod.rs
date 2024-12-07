use crate::ops::{OnnxOpError, OnnxOpRegistry};
mod basics;
mod clip;
mod cmp;
mod matmul;
mod sqrt;

pub(crate) fn register(registry: &mut OnnxOpRegistry) -> Result<(), OnnxOpError> {
    //comparison
    registry.insert("Equal", Box::new(cmp::Equal))?;

    //basics
    registry.insert("Add", Box::new(basics::Add))?;
    registry.insert("Sub", Box::new(basics::Sub))?;
    registry.insert("Mul", Box::new(basics::Mul))?;
    registry.insert("Div", Box::new(basics::Div))?;
    registry.insert("Exp", Box::new(basics::Exp))?;
    registry.insert("Pow", Box::new(basics::Pow))?;
    registry.insert("Sign", Box::new(basics::Sign))?;
    registry.insert("Clip", Box::new(clip::Clip))?;
    registry.insert("Sqrt", Box::new(sqrt::Sqrt))?;

    //matrix
    registry.insert("MatMul", Box::new(matmul::MatMul))?;

    Ok(())
}
