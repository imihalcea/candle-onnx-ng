use crate::ops::ComputeNode;
use candle_core as candle;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};

pub enum OpOutput {
    Single(String, candle::Tensor),
    Multiple(Vec<(String, candle::Tensor)>),
    Branch(String),
}

#[derive(Debug, PartialEq, Eq)]
pub enum OnnxOpError {
    InvalidInput(String),
    InvalidOutput(String),
    InvalidAttribute(String),
    ComputationFailed(String),
    UnsupportedOp(String),
    DuplicateOp(String),
    MalformedOp(String),
    UnsupportedType(String),
    UnsupportedAttribute(String),
}

impl From<OnnxOpError> for candle::Error {
    fn from(e: OnnxOpError) -> Self {
        candle::Error::Msg(format!("{:?}", e))
    }
}

impl From<candle_core::Error> for OnnxOpError {
    fn from(e: candle_core::Error) -> Self {
        OnnxOpError::ComputationFailed(format!("{:?}", e))
    }
}

impl Display for OnnxOpError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            OnnxOpError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
            OnnxOpError::InvalidAttribute(s) => write!(f, "Invalid attribute: {}", s),
            OnnxOpError::InvalidOutput(s) => write!(f, "Invalid output: {}", s),
            OnnxOpError::ComputationFailed(s) => write!(f, "Computation failed: {}", s),
            OnnxOpError::UnsupportedOp(s) => write!(f, "Unsupported op: {}", s),
            OnnxOpError::MalformedOp(s) => write!(f, "Malformed op: {}", s),
            OnnxOpError::DuplicateOp(s) => write!(f, "Duplicate op: {}", s),
            OnnxOpError::UnsupportedType(s) => write!(f, "Unsupported type: {}", s),
            OnnxOpError::UnsupportedAttribute(s) => write!(f, "Unsupported attribute: {}", s),
        }
    }
}

pub trait OnnxOp: Send + Sync{
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError>;
}

#[derive(Default)]
pub struct OnnxOpRegistry {
    ops: HashMap<String, Box<dyn OnnxOp>>,
}

impl OnnxOpRegistry {
    pub fn new() -> Self {
        Self {
            ops: HashMap::new(),
        }
    }
    pub fn insert(&mut self, name: &str, op: Box<dyn OnnxOp>) -> Result<(), OnnxOpError> {
        match self.ops.entry(name.to_string()) {
            Entry::Vacant(vacant_entry) => {
                vacant_entry.insert(op);
                Ok(())
            }
            Entry::Occupied(_) => Err(OnnxOpError::DuplicateOp(name.to_string())),
        }
    }

    pub fn get(&self, name: &str) -> Result<&dyn OnnxOp, OnnxOpError> {
        match self.ops.get(name) {
            Some(op) => Ok(op.as_ref()),
            None => Err(OnnxOpError::UnsupportedOp(name.to_string())),
        }
    }
}

#[cfg(test)]
mod onnxop_registry_tests {
    use super::*;
    use candle::Device;
    #[test]
    fn nominal_case() {
        //Given
        let dummy_op = Box::new(DummyOp);
        let mut registry = OnnxOpRegistry::new();

        //When
        registry.insert("DummyOp", dummy_op).unwrap();
        let op = registry.get("DummyOp");

        //Then
        assert!(op.is_ok());
    }

    #[test]
    fn unsupported_op() {
        //Given
        let registry = OnnxOpRegistry::new();

        //When
        let op = registry.get("Foo");

        //Then
        match op {
            Err(OnnxOpError::UnsupportedOp(_)) => {}
            _ => panic!("Expected unsupported op error"),
        }
    }

    #[test]
    fn duplicate_op() {
        //Given
        let dummy_op = Box::new(DummyOp);
        let mut registry = OnnxOpRegistry::new();
        registry.insert("DummyOp", dummy_op).unwrap();

        //When
        let dummy_op = Box::new(DummyOp);
        let result = registry.insert("DummyOp", dummy_op);

        //Then
        match result {
            Err(OnnxOpError::DuplicateOp(_)) => {}
            _ => panic!("Expected duplicate op error"),
        }
    }

    struct DummyOp;
    impl OnnxOp for DummyOp {
        fn eval(&self, _node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
            Ok(OpOutput::Single(
                "dummy".to_string(),
                candle::Tensor::new(vec![1u8, 1], &Device::Cpu)?,
            ))
        }
    }
}
