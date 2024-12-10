use crate::ops::{OnnxOpError, OnnxOpRegistry};
mod avgpool;
mod batchnorm;
mod conv;
mod dropout;
mod maxpool;
mod softmax;
mod basics;

pub(crate) fn register(registry: &mut OnnxOpRegistry) -> Result<(), OnnxOpError> {
    registry.insert("LogSoftmax", Box::new(softmax::LogSoftmax))?;
    registry.insert("Softmax", Box::new(softmax::Softmax))?;

    registry.insert("Dropout", Box::new(dropout::Dropout))?;
    registry.insert("AveragePool", Box::new(avgpool::AveragePool))?;
    registry.insert("MaxPool", Box::new(maxpool::MaxPool))?;

    registry.insert(
        "BatchNormalization",
        Box::new(batchnorm::BatchNormalization),
    )?;
    registry.insert("Conv", Box::new(conv::Conv))?;

    registry.insert("Sigmoid", Box::new(basics::Sigmoid))?;

    Ok(())
}
