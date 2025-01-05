mod lstm;

use crate::ops::{OnnxOpError, OnnxOpRegistry};

pub(crate) fn register(registry: &mut OnnxOpRegistry) -> Result<(), OnnxOpError> {
    registry.insert("LSTM", Box::new(lstm::Lstm))?;
    Ok(())
}

