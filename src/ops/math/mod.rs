use crate::ops::{OnnxOpError, OnnxOpRegistry};
mod basics;
mod clip;
mod cmp;
mod cumsum;
mod gemm;
mod matmul;
mod sqrt;

pub(crate) fn register(registry: &mut OnnxOpRegistry) -> Result<(), OnnxOpError> {
    //comparison
    registry.insert("Equal", Box::new(cmp::Equal))?;
    registry.insert("Greater", Box::new(cmp::Greater))?;
    registry.insert("Less", Box::new(cmp::Less))?;

    //basics
    registry.insert("Identity", Box::new(basics::Identity))?;
    registry.insert("Abs", Box::new(basics::Abs))?;
    registry.insert("Add", Box::new(basics::Add))?;
    registry.insert("Sub", Box::new(basics::Sub))?;
    registry.insert("Mul", Box::new(basics::Mul))?;
    registry.insert("Div", Box::new(basics::Div))?;
    registry.insert("Exp", Box::new(basics::Exp))?;
    registry.insert("Pow", Box::new(basics::Pow))?;
    registry.insert("Sign", Box::new(basics::Sign))?;
    registry.insert("Clip", Box::new(clip::Clip))?;
    registry.insert("Sqrt", Box::new(sqrt::Sqrt))?;
    registry.insert("Log", Box::new(basics::Log))?;
    registry.insert("Cos", Box::new(basics::Cos))?;
    registry.insert("Sin", Box::new(basics::Sin))?;
    registry.insert("Neg", Box::new(basics::Neg))?;
    registry.insert("Erf", Box::new(basics::Erf))?;
    registry.insert("Tanh", Box::new(basics::Tanh))?;
    registry.insert("Ceil", Box::new(basics::Ceil))?;
    registry.insert("Floor", Box::new(basics::Floor))?;
    registry.insert("CumSum", Box::new(cumsum::CumSum))?;
    registry.insert("Gemm", Box::new(gemm::Gemm))?;

    //matrix
    registry.insert("MatMul", Box::new(matmul::MatMul))?;

    //statistics
    registry.insert("Min", Box::new(basics::Min))?;
    registry.insert("Where", Box::new(basics::Where))?;

    Ok(())
}
