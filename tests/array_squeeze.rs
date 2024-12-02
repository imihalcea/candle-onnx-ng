use candle_core::{Device, Tensor};
use candle_onnx_ng::onnx::{GraphProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

mod utils;

#[test]
fn test_squeeze_with_no_axes_operation() -> candle_core::Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Squeeze".to_string(),
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
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        &[1, 1, 2, 2],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("INPUT_X".to_string(), x.clone());

    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let result = eval.get("OUTPUT_Z").expect("Output 'z' not found");

    assert_eq!(result.shape().dims(), &[1, 2, 2]);
    assert_eq!(
        result.to_vec3::<f32>()?,
        vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]
    );

    Ok(())
}

#[test]
fn test_squeeze_axes_0_operation() -> candle_core::Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Squeeze".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec!["INPUT_X".to_string(), "axes".to_string()],
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
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        &[1, 1, 2, 2],
        &Device::Cpu,
    )?;

    let axes = Tensor::from_vec(vec![0i64], &[1], &Device::Cpu)?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("INPUT_X".to_string(), x.clone());
    inputs.insert("axes".to_string(), axes.clone());

    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let result = eval.get("OUTPUT_Z").expect("Output 'z' not found");

    assert_eq!(result.shape().dims(), &[1, 2, 2]);

    Ok(())
}

#[test]
fn test_squeeze_neg_axes_operation() -> candle_core::Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Squeeze".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec!["INPUT_X".to_string(), "axes".to_string()],
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

    let x = Tensor::rand(1f32, 100f32, &[1, 3, 1, 5], &Device::Cpu)?;

    let axes = Tensor::from_vec(vec![-2i64], &[1], &Device::Cpu)?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("INPUT_X".to_string(), x.clone());
    inputs.insert("axes".to_string(), axes.clone());

    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let result = eval.get("OUTPUT_Z").expect("Output 'z' not found");

    assert_eq!(result.shape().dims(), &[1, 3, 5]);

    Ok(())
}
