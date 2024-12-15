use candle_core::{Device, Tensor};
use candle_onnx_ng::onnx::{GraphProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

mod utils;
#[test]
fn test_slice() -> candle_core::Result<()> {
    let model = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Slice".to_string(),
            input: vec![
                "data".to_string(),
                "starts".to_string(),
                "ends".to_string(),
                "axes".to_string(),
                "steps".to_string(),
            ],
            output: vec!["result".to_string()],
            ..NodeProto::default()
        }],
        input: ["data", "starts", "ends", "axes", "steps"]
            .into_iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                r#type: None,
                doc_string: "".to_string(),
            })
            .collect(),
        output: ["result"]
            .into_iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                r#type: None,
                doc_string: "".to_string(),
            })
            .collect(),
        ..GraphProto::default()
    }));

    /*
    data = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    axes = [0, 1]
    starts = [1, 0]
    ends = [2, 3]
    steps = [1, 2]
    result = [
        [5, 7],
    ]
    */

    let outputs = simple_eval(
        &model,
        HashMap::from_iter([
            (
                "data".to_string(),
                Tensor::from_vec(vec![1i64, 2, 3, 4, 5, 6, 7, 8], (2, 4), &Device::Cpu)?,
            ),
            (
                "starts".to_string(),
                Tensor::from_vec(vec![1i64, 0], (2,), &Device::Cpu)?,
            ),
            (
                "ends".to_string(),
                Tensor::from_vec(vec![2i64, 3], (2,), &Device::Cpu)?,
            ),
            (
                "axes".to_string(),
                Tensor::from_vec(vec![0i64, 1], (2,), &Device::Cpu)?,
            ),
            (
                "steps".to_string(),
                Tensor::from_vec(vec![1i64, 2], (2,), &Device::Cpu)?,
            ),
        ]),
    )?;
    let actual = outputs.get("result").unwrap().to_vec2::<i64>()?;
    assert_eq!(actual, vec![vec![5i64, 7]]);

    /*
    data = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    starts = [0, 1]
    ends = [-1, 1000]
    result = [
        [2, 3, 4],
    ]
    */
    let model = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Slice".to_string(),
            input: vec!["data".to_string(), "starts".to_string(), "ends".to_string()],
            output: vec!["result".to_string()],
            ..NodeProto::default()
        }],
        input: ["data", "starts", "ends"]
            .into_iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                r#type: None,
                doc_string: "".to_string(),
            })
            .collect(),
        output: ["result"]
            .into_iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                r#type: None,
                doc_string: "".to_string(),
            })
            .collect(),
        ..GraphProto::default()
    }));
    let outputs = simple_eval(
        &model,
        HashMap::from_iter([
            (
                "data".to_string(),
                Tensor::from_vec(vec![1i64, 2, 3, 4, 5, 6, 7, 8], (2, 4), &Device::Cpu)?,
            ),
            (
                "starts".to_string(),
                Tensor::from_vec(vec![0i64, 1], (2,), &Device::Cpu)?,
            ),
            (
                "ends".to_string(),
                Tensor::from_vec(vec![-1i64, 1000], (2,), &Device::Cpu)?,
            ),
        ]),
    )?;
    let actual = outputs.get("result").unwrap().to_vec2::<i64>()?;
    assert_eq!(actual, vec![vec![2i64, 3, 4]]);

    Ok(())
}
