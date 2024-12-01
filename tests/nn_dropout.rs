use std::collections::HashMap;
use candle_core::{Device, Tensor};
use candle_onnx_ng::onnx::{GraphProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;

mod utils;

// "Dropout"
#[test]
fn test_dropout_operation() -> candle_core::Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Dropout".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec!["INPUT_X".to_string()],
            output: vec!["OUTPUT_Z".to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: "INPUT_X".to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: "INPUT_Y".to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
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
    let x = Tensor::from_vec(
        //
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        &[2, 2],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("INPUT_X".to_string(), x);

    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(results, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    Ok(())
}