use candle_core::{Device, Tensor};
use candle_onnx_ng::onnx::attribute_proto::AttributeType;
use candle_onnx_ng::onnx::{AttributeProto, GraphProto, ModelProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

mod utils;
#[test]
fn test_cumsum_1d() -> candle_core::Result<()> {
    let manual_graph = create_cumsum_graph(0, 0);
    let x = Tensor::new(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32], &Device::Cpu)?;
    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("INPUT_X".to_string(), x);
    inputs.insert("axis".to_string(), Tensor::new(0i64, &Device::Cpu)?);

    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");

    let results = z.to_vec1::<f32>()?;
    assert_eq!(results, vec![1.0, 3.0, 6.0, 10.0]);

    Ok(())
}

#[test]
fn test_cumsum_2d_axis_0() -> candle_core::Result<()> {
    let manual_graph = create_cumsum_graph(0, 0);
    let x = Tensor::from_vec(
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32],
        (2, 3),
        &Device::Cpu,
    )?;
    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("INPUT_X".to_string(), x);
    inputs.insert("axis".to_string(), Tensor::new(0i64, &Device::Cpu)?);

    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;
    assert_eq!(results, vec![vec![1.0, 2.0, 3.0], vec![5.0, 7.0, 9.0]]);

    Ok(())
}

#[test]
fn test_cumsum_2d_axis_1() -> candle_core::Result<()> {
    let manual_graph = create_cumsum_graph(0, 0);
    let x = Tensor::from_vec(
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32],
        (2, 3),
        &Device::Cpu,
    )?;
    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("INPUT_X".to_string(), x);
    inputs.insert("axis".to_string(), Tensor::new(1i64, &Device::Cpu)?);

    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;
    assert_eq!(results, vec![vec![1.0, 3.0, 6.0], vec![4.0, 9.0, 15.0]]);

    Ok(())
}

fn create_cumsum_graph(exclusive_value: i64, reverse_value: i64) -> ModelProto {
    let exclusive = AttributeProto {
        name: "exclusive".to_string(),
        ref_attr_name: "exclusive".to_string(),
        i: exclusive_value,
        doc_string: "exclusive".to_string(),
        r#type: AttributeType::Int.into(),
        f: 0.0,
        s: vec![],
        t: None,
        g: None,
        sparse_tensor: None,
        tp: None,
        floats: vec![],
        ints: vec![],
        strings: vec![],
        tensors: vec![],
        graphs: vec![],
        sparse_tensors: vec![],
        type_protos: vec![],
    };

    let reverse = AttributeProto {
        name: "reverse".to_string(),
        ref_attr_name: "reverse".to_string(),
        i: reverse_value,
        doc_string: "reverse".to_string(),
        r#type: AttributeType::Int.into(),
        f: 0.0,
        s: vec![],
        t: None,
        g: None,
        sparse_tensor: None,
        tp: None,
        floats: vec![],
        ints: vec![],
        strings: vec![],
        tensors: vec![],
        graphs: vec![],
        sparse_tensors: vec![],
        type_protos: vec![],
    };

    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "CumSum".to_string(),
            domain: "".to_string(),
            attribute: vec![exclusive, reverse],
            input: vec!["INPUT_X".to_string(), "axis".to_string()],
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

    manual_graph
}
