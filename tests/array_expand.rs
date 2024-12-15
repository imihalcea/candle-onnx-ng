use candle_core::{Device, Tensor};
use candle_onnx_ng::onnx::{AttributeProto, GraphProto, ModelProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

mod utils;
#[test]
fn test_expand_dim_changed() -> candle_core::Result<()> {
    // Create a manual graph for the Expand operation
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Expand".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec!["data".to_string(), "new_shape".to_string()],
            output: vec!["expanded".to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        input: vec![
            ValueInfoProto {
                name: "data".to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: "new_shape".to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: "expanded".to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        ..GraphProto::default()
    }));

    // Input tensor with shape [3, 1]
    let data = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32], (3, 1), &Device::Cpu)?;

    // New shape tensor: [2, 1, 6]
    let new_shape = Tensor::from_vec(vec![2i64, 1, 6], (3,), &Device::Cpu)?;

    // Expected output after expansion
    let expected = Tensor::from_vec(
        vec![
            1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32,
            2.0f32, 3.0f32, 3.0f32, 3.0f32, 3.0f32, 3.0f32, 3.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32,
            1.0f32, 1.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 3.0f32, 3.0f32, 3.0f32,
            3.0f32, 3.0f32, 3.0f32,
        ],
        (2, 3, 6),
        &Device::Cpu,
    )?;

    // Execute the model evaluation
    let inputs = HashMap::from_iter([
        ("data".to_string(), data),
        ("new_shape".to_string(), new_shape),
    ]);
    let result = simple_eval(&manual_graph, inputs)?;

    // Retrieve and compare the result
    let expanded = result.get("expanded").expect("Output 'expanded' not found");

    assert_eq!(expanded.to_vec3::<f32>()?, expected.to_vec3::<f32>()?);

    Ok(())
}

#[test]
fn test_expand_dim_unchanged() -> candle_core::Result<()> {
    // Create a manual graph for the Expand operation
    let manual_graph = make_graph_helper("Expand", &["data", "new_shape"], &["expanded"], vec![]);

    // Input tensor with shape [3, 1] and dtype f32
    let data = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32], (3, 1), &Device::Cpu)?;

    // New shape tensor: [3, 4]
    let new_shape = Tensor::from_vec(vec![3i64, 4], (2,), &Device::Cpu)?;

    // Expected output after expansion, dtype f32
    let expected = Tensor::from_vec(
        vec![
            1.0f32, 1.0f32, 1.0f32, 1.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 3.0f32, 3.0f32, 3.0f32,
            3.0f32,
        ],
        (3, 4),
        &Device::Cpu,
    )?;

    // Execute the model evaluation
    let inputs = HashMap::from_iter([
        ("data".to_string(), data),
        ("new_shape".to_string(), new_shape),
    ]);
    let result = simple_eval(&manual_graph, inputs)?;

    // Retrieve and compare the result
    let expanded = result.get("expanded").expect("Output 'expanded' not found");
    assert_eq!(expanded.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);

    Ok(())
}

fn make_graph_helper(
    op_name: &str,
    inputs: &[&str],
    outputs: &[&str],
    attribs: Vec<AttributeProto>,
) -> ModelProto {
    utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: op_name.to_string(),
            domain: "".to_string(),
            attribute: attribs,
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        input: inputs
            .iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                ..ValueInfoProto::default()
            })
            .collect(),
        output: outputs
            .iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                ..ValueInfoProto::default()
            })
            .collect(),
        ..GraphProto::default()
    }))
}
