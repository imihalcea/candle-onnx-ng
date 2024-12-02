use crate::ops::{OnnxOpError, OnnxOpRegistry};

mod constshape;
mod flatten;
mod reshape;
mod squeeze;
mod transpose;

pub(crate) fn register(registry: &mut OnnxOpRegistry) -> Result<(), OnnxOpError> {
    registry.insert("Reshape", Box::new(reshape::Reshape))?;
    registry.insert("Transpose", Box::new(transpose::Transpose))?;
    registry.insert("Flatten", Box::new(flatten::Flatten))?;
    registry.insert("Squeeze", Box::new(squeeze::Squeeze))?;
    registry.insert("ConstantOfShape", Box::new(constshape::ConstantOfShape))?;
    Ok(())
}
