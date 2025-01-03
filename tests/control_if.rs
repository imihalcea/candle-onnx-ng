use candle_core::{bail, Device, Tensor};
use candle_onnx_ng::onnx::attribute_proto::AttributeType;
use candle_onnx_ng::onnx::tensor_proto::DataType;
use candle_onnx_ng::onnx::tensor_shape_proto::{dimension, Dimension};
use candle_onnx_ng::onnx::{
    type_proto, AttributeProto, GraphProto, NodeProto, TensorProto, TensorShapeProto, TypeProto,
    ValueInfoProto,
};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

mod utils;

// "If"
#[test]
fn test_if() -> candle_core::Result<()> {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let output_type_proto = Some(TypeProto {
        value: Some(type_proto::Value::TensorType(type_proto::Tensor {
            elem_type: DataType::Float.into(),
            shape: Some(TensorShapeProto {
                dim: vec![Dimension {
                    denotation: "".to_string(),
                    value: Some(dimension::Value::DimValue(5)),
                }],
            }),
        })),
        denotation: "".to_string(),
    });
    let then_branch = GraphProto {
        output: vec![ValueInfoProto {
            name: "then_out".to_string(),
            r#type: output_type_proto.clone(),
            doc_string: "".to_string(),
        }],
        node: vec![NodeProto {
            op_type: "Constant".to_string(),
            input: vec![],
            output: vec!["then_out".to_string()],
            attribute: vec![AttributeProto {
                name: "value".to_string(),
                r#type: AttributeType::Tensor.into(),
                t: Some(TensorProto {
                    dims: vec![x.len() as i64],
                    float_data: x.clone(),
                    data_type: DataType::Float.into(),
                    ..TensorProto::default()
                }),
                ..AttributeProto::default()
            }],
            ..NodeProto::default()
        }],
        ..GraphProto::default()
    };
    let else_branch = GraphProto {
        output: vec![ValueInfoProto {
            name: "else_out".to_string(),
            r#type: output_type_proto.clone(),
            doc_string: "".to_string(),
        }],
        node: vec![NodeProto {
            op_type: "Constant".to_string(),
            input: vec![],
            output: vec!["else_out".to_string()],
            attribute: vec![AttributeProto {
                name: "value".to_string(),
                r#type: AttributeType::Tensor.into(),
                t: Some(TensorProto {
                    dims: vec![y.len() as i64],
                    float_data: y.clone(),
                    data_type: DataType::Float.into(),
                    ..TensorProto::default()
                }),
                ..AttributeProto::default()
            }],
            ..NodeProto::default()
        }],
        ..GraphProto::default()
    };
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "If".to_string(),
            attribute: vec![
                AttributeProto {
                    name: "then_branch".to_string(),
                    r#type: AttributeType::Graph.into(),
                    g: Some(then_branch),
                    ..AttributeProto::default()
                },
                AttributeProto {
                    name: "else_branch".to_string(),
                    r#type: AttributeType::Graph.into(),
                    g: Some(else_branch),
                    ..AttributeProto::default()
                },
            ],
            input: vec!["cond".to_string()],
            output: vec!["res".to_string()],
            ..NodeProto::default()
        }],
        input: vec![],
        output: vec![ValueInfoProto {
            name: "res".to_string(),
            doc_string: "".to_string(),
            r#type: output_type_proto.clone(),
        }],
        ..GraphProto::default()
    }));

    for cond in [1u8, 0] {
        let inputs =
            HashMap::from_iter([("cond".to_string(), Tensor::full(cond, (1,), &Device::Cpu)?)]);
        let outputs = simple_eval(&manual_graph, inputs)?;
        let expected = if cond != 0 { &x } else { &y };
        let Some(res) = outputs.get("res") else {
            bail!("outputs didn't contain expected key `res`: {outputs:?}");
        };
        assert_eq!(&res.to_vec1::<f32>()?, expected);
    }
    Ok(())
}
