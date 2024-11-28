use std::collections::HashMap;
use candle_core::{Device, Tensor};
use candle_onnx_ng::onnx::{GraphProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;

pub mod utils;

// "Div"
#[test]
fn test_div_operation() -> candle_core::Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Div".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec!["INPUT_X".to_string(), "INPUT_Y".to_string()],
            output: vec!["OUTPUT_Z".to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: "OUTPUT_Z".to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("INPUT_X".to_string(), Tensor::new(&[2.], &Device::Cpu)?);
    inputs.insert("INPUT_Y".to_string(), Tensor::new(&[2.], &Device::Cpu)?);

    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");
    let first = z.to_vec1::<f64>()?[0];
    assert_eq!(first, 1.0f64);
    Ok(())
}