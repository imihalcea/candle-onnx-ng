use crate::onnx::attribute_proto::AttributeType;
use crate::onnx::tensor_proto::DataType;
use crate::onnx::NodeProto;
use crate::onnx::{self, GraphProto};
use crate::ops::registry;
use crate::ops::OnnxOpError;
use crate::parser;
use crate::parser::Value;
use candle_core as candle;
use candle_core::{bail, DType, Device, Tensor};
use std::collections::{HashMap, HashSet};

//This struct is used to represent a node in the computation graph
//The idea is not to use the NodeProto directly in the computation graph
//On a longer term, this can lead to a more optimized representation of the computation graph.
//For now, it is just a wrapper around the NodeProto and the context
pub struct ComputeNode<'a> {
    node_proto: &'a NodeProto,
    context: &'a HashMap<String, Tensor>,
}

impl<'a> ComputeNode<'a> {
    pub fn new(node_proto: &'a NodeProto, context: &'a HashMap<String, Tensor>) -> Self {
        ComputeNode {
            node_proto,
            context,
        }
    }

    pub fn input_len(&self) -> usize {
        self.node_proto.input.len()
    }

    pub fn get_input(&self, index: usize) -> Result<&Tensor, OnnxOpError> {
        let input_name = self
            .node_proto
            .input
            .get(index)
            .ok_or_else(|| OnnxOpError::InvalidInput(format!("input {} not found", index)))?;

        self.context
            .get(input_name)
            .ok_or_else(|| OnnxOpError::InvalidInput(format!("input {} not found", index)))
    }

    pub fn get_output(&self, index: usize) -> Result<&String, OnnxOpError> {
        self.node_proto
            .output
            .get(index)
            .ok_or_else(|| OnnxOpError::InvalidOutput(format!("output {} not found", index)))
    }

    pub(crate) fn get_attr_opt<T: parser::Attr + ?Sized>(
        &self,
        name: &str,
    ) -> Result<Option<&T>, OnnxOpError> {
        parser::get_attr_opt(&self.node_proto, name).map_err(OnnxOpError::from)
    }

    pub(crate) fn get_attr<T: parser::Attr + ?Sized>(&self, name: &str) -> Result<&T, OnnxOpError> {
        parser::get_attr(&self.node_proto, name).map_err(OnnxOpError::from)
    }
    pub(crate) fn get_attr_opt_owned<T: parser::AttrOwned>(
        &self,
        name: &str,
    ) -> Result<Option<T>, OnnxOpError> {
        parser::get_attr_opt_owned(&self.node_proto, name).map_err(OnnxOpError::from)
    }
}
