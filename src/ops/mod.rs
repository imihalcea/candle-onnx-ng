pub mod onnxop;
pub use onnxop::{OnnxOp, OnnxOpError, OnnxOpRegistry, OpOutput};

pub mod compute_node;
pub use compute_node::ComputeNode;

mod array;
mod logic;
mod math;
mod nn;

pub fn registry() -> Result<OnnxOpRegistry, OnnxOpError> {
    let mut registry = OnnxOpRegistry::new();

    logic::register(&mut registry)?;
    math::register(&mut registry)?;
    array::register(&mut registry)?;
    nn::register(&mut registry)?;
    Ok(registry)
}
