use candle_core::{Device, Tensor};
use candle_onnx_ng::onnx::{GraphProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

mod utils;
#[test]
fn test_unsqueeze() -> candle_core::Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Unsqueeze".to_string(),
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
        value_info: vec![ValueInfoProto {
            name: "INPUT_X".to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(
        vec![
            1.0f32, 2.0f32, //
            3.0f32, 4.0f32, //
        ],
        &[2, 2],
        &Device::Cpu,
    )?;
    let y = Tensor::from_vec(vec![-1i64], &[1], &Device::Cpu)?;

    let inputs = HashMap::from_iter([
        ("INPUT_X".to_string(), x.clone()),
        ("INPUT_Y".to_string(), y),
    ]);

    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");
    assert_eq!(z.dims(), &[2, 2, 1]);
    assert_eq!(
        z.flatten_all()?.to_vec1::<f32>()?,
        x.flatten_all()?.to_vec1::<f32>()?
    );

    Ok(())
}
