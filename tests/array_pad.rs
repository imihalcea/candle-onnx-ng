use std::collections::HashMap;
use candle_core::{Device, Tensor};
use candle_onnx_ng::onnx::{AttributeProto, GraphProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::onnx::attribute_proto::AttributeType;
use candle_onnx_ng::simple_eval;

mod utils;
#[test]
fn test_pad() -> candle_core::Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#pad
    let data = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
        ],
        (2, 3),
        &Device::Cpu,
    )?;
    let pads = Tensor::from_vec(vec![0i64, 1, 0, 0], (4,), &Device::Cpu)?;
    let mode = "reflect";

    let expected = Tensor::from_vec(
        vec![
            2.0, 1.0, 2.0, 3.0, //
            5.0, 4.0, 5.0, 6.0, //
        ],
        (2, 4),
        &Device::Cpu,
    )?;

    let model = utils::create_model_proto_with_graph(Some(GraphProto {
        input: vec![
            ValueInfoProto {
                name: "data".to_string(),
                ..ValueInfoProto::default()
            },
            ValueInfoProto {
                name: "pads".to_string(),
                ..ValueInfoProto::default()
            },
        ],
        output: vec![ValueInfoProto {
            name: "output".to_string(),
            ..ValueInfoProto::default()
        }],
        node: vec![NodeProto {
            op_type: "Pad".to_string(),
            input: vec!["data".to_string(), "pads".to_string()],
            output: vec!["output".to_string()],
            attribute: vec![AttributeProto {
                name: "mode".to_string(),
                r#type: AttributeType::String.into(),
                s: mode.as_bytes().to_vec(),
                ..AttributeProto::default()
            }],
            ..NodeProto::default()
        }],
        ..GraphProto::default()
    }));

    let inputs = HashMap::from_iter([("data".to_string(), data), ("pads".to_string(), pads)]);
    let res = simple_eval(&model, inputs)?;
    let Some(actual) = res.get("output") else {
        candle_core::bail!("outputs didn't contain expected key `output`: {res:?}");
    };

    assert_eq!(actual.to_vec2::<f64>()?, expected.to_vec2::<f64>()?);
    Ok(())
}