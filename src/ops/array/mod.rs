use crate::ops::{OnnxOpError, OnnxOpRegistry};

mod arg;
mod cast;
mod concat;
mod constant;
mod constshape;
mod expand;
mod flatten;
mod gather;
mod pad;
mod random;
mod range;
mod reshape;
mod shape;
mod size;
mod slice;
mod split;
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
    registry.insert("Concat", Box::new(concat::Concat))?;
    registry.insert("Constant", Box::new(constant::Constant))?;
    registry.insert("Cast", Box::new(cast::Cast))?;
    registry.insert("Pad", Box::new(pad::Pad))?;
    registry.insert("Slice", Box::new(slice::Slice))?;
    registry.insert("Expand", Box::new(expand::Expand))?;
    registry.insert("ArgMin", Box::new(arg::ArgMin))?;
    registry.insert("ArgMax", Box::new(arg::ArgMax))?;
    registry.insert("Split", Box::new(split::Split))?;
    registry.insert("RandomNormal", Box::new(random::RandomNormal))?;
    registry.insert("RandomUniform", Box::new(random::RandomUniform))?;
    Ok(())
}
