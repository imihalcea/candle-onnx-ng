mod r#if;

use crate::ops::{OnnxOpError, OnnxOpRegistry};

//only if is implemented for the moment
//loops will be implemented in the future
pub(crate) fn register(registry: &mut OnnxOpRegistry) -> Result<(), OnnxOpError> {
    registry.insert("If", Box::new(r#if::If))?;
    Ok(())
}
