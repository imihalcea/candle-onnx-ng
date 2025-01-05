pub mod onnxop;
pub use onnxop::{OnnxOp, OnnxOpError, OnnxOpRegistry, OpOutput};

pub mod compute_node;
pub use compute_node::ComputeNode;

mod array;
mod control;
mod logic;
mod math;
mod nn;
pub mod tensor_helper;
mod rnn;

pub fn registry() -> Result<OnnxOpRegistry, OnnxOpError> {
    let mut registry = OnnxOpRegistry::new();

    logic::register(&mut registry)?;
    math::register(&mut registry)?;
    array::register(&mut registry)?;
    nn::register(&mut registry)?;
    control::register(&mut registry)?;
    rnn::register(&mut registry)?;
    Ok(registry)
}
