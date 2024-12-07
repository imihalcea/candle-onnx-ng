use crate::ops::{OnnxOpError, OnnxOpRegistry};

mod constshape;
mod flatten;
mod gather;
mod range;
mod reshape;
mod shape;
mod size;
mod squeeze;
mod transpose;
mod unsqueeze;

pub(crate) fn register(registry: &mut OnnxOpRegistry) -> Result<(), OnnxOpError> {
    registry.insert("Reshape", Box::new(reshape::Reshape))?;
    registry.insert("Transpose", Box::new(transpose::Transpose))?;
    registry.insert("Flatten", Box::new(flatten::Flatten))?;
    registry.insert("Squeeze", Box::new(squeeze::Squeeze))?;
    registry.insert("ConstantOfShape", Box::new(constshape::ConstantOfShape))?;
    registry.insert("Unsqueeze", Box::new(unsqueeze::Unsqueeze))?;
    registry.insert("Gather", Box::new(gather::Gather))?;
    registry.insert("GatherElements", Box::new(gather::GatherElements))?;
    registry.insert("Shape", Box::new(shape::Shape))?;
    registry.insert("Size", Box::new(size::Size))?;
    registry.insert("Range", Box::new(range::Range))?;
    Ok(())
}
