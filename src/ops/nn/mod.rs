use crate::ops::{OnnxOpError, OnnxOpRegistry};
mod avgpool;
mod basics;
mod batchnorm;
mod conv;
mod dropout;
mod maxpool;
mod reduce;
mod softmax;

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
    registry.insert("Gelu", Box::new(basics::Gelu))?;
    registry.insert("Relu", Box::new(basics::Relu))?;
    registry.insert("LeakyRelu", Box::new(basics::LeakyRelu))?;
    registry.insert("ReduceMin", Box::new(reduce::ReduceMin))?;
    registry.insert("ReduceMax", Box::new(reduce::ReduceMax))?;
    registry.insert("ReduceMean", Box::new(reduce::ReduceMean))?;
    registry.insert("ReduceSum", Box::new(reduce::ReduceSum))?;
    registry.insert("ReduceL2", Box::new(reduce::ReduceL2))?;

    Ok(())
}
