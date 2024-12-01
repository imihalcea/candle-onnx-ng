use candle_core::{Device, Tensor};
use candle_onnx_ng::onnx::attribute_proto::AttributeType;
use candle_onnx_ng::onnx::{AttributeProto, GraphProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

mod utils;

// MaxPool

#[test]
fn test_maxpool_1d_default() -> candle_core::Result<()> {
    //TO DO test example https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool
    Ok(())
}
