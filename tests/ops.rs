use candle_core::{Result, Tensor};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

pub mod utils;

#[test]
fn test_evaluation_fails_without_defined_graph() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(None);
    let inputs: HashMap<String, Tensor> = HashMap::new();
    let result = simple_eval(&manual_graph, inputs);
    match result {
        Err(err) => assert!(err.to_string().contains("no graph defined in proto")),
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
