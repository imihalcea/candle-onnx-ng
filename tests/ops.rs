use candle_core::test_utils::to_vec2_round;
use candle_core::{bail, DType, Device, NdArray, Result, Tensor};
use candle_onnx_ng::onnx::attribute_proto::AttributeType;
use candle_onnx_ng::onnx::tensor_proto::DataType;
use candle_onnx_ng::onnx::tensor_shape_proto::{dimension, Dimension};
use candle_onnx_ng::onnx::{type_proto, TensorProto, TensorShapeProto, TypeProto};
use candle_onnx_ng::onnx::{AttributeProto, GraphProto, ModelProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

pub mod utils;

const INPUT_X: &str = "x";
const INPUT_Y: &str = "y";
const INPUT_A: &str = "a";
const OUTPUT_Z: &str = "z";

#[test]
fn test_evaluation_fails_without_defined_graph() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(None);
    let inputs: HashMap<String, Tensor> = HashMap::new();
    match simple_eval(&manual_graph, inputs) {
        Err(err) => assert_eq!(err.to_string(), "no graph defined in proto"),
        Ok(_) => panic!("Expected an error due to undefined graph"),
    }
    Ok(())
}

// Below are new_ops that are implemented but not tested yet

// "MaxPool"
// #[test]

// "AveragePool"
// #[test]

// "BatchNormalization"
// #[test]

// "Squeeze"
// #[test]

// "ConstantOfShape"

// "Unsqueeze"

// "Clip"
// #[test]

// "Shape"

// "Conv"
// #[test]

// "Concat"
// #[test]

// "Erf"
// #[test]

// "Constant"
// #[test]

// "Cast"
// #[test]

