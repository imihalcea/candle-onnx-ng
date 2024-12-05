use candle_core::{Device, Tensor};
use candle_onnx_ng::onnx::{GraphProto, ModelProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

pub mod utils;

#[test]
fn test_clip_min_max_provided() -> candle_core::Result<()> {
    let manual_graph = create_model();

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(
        "INPUT_X".to_string(),
        Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &Device::Cpu)?,
    );
    inputs.insert("MIN".to_string(), Tensor::new(&[2.0f32], &Device::Cpu)?);
    inputs.insert("MAX".to_string(), Tensor::new(&[4.0f32], &Device::Cpu)?);

    let eval = simple_eval(&manual_graph, inputs)?;

    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");
    assert_eq!(z.to_vec1::<f32>()?, vec![2.0f32, 2.0, 3.0, 4.0, 4.0]);

    Ok(())
}

#[test]
fn test_clip_min_greater_than_max() -> candle_core::Result<()> {
    let manual_graph = create_model();

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(
        "INPUT_X".to_string(),
        Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?,
    );
    inputs.insert("MIN".to_string(), Tensor::new(&[5.0f32], &Device::Cpu)?);
    inputs.insert("MAX".to_string(), Tensor::new(&[3.0f32], &Device::Cpu)?);

    let eval = simple_eval(&manual_graph, inputs)?;

    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");
    assert_eq!(z.to_vec1::<f32>()?, vec![3.0f32, 3.0, 3.0]);

    Ok(())
}

#[test]
fn test_no_clipping_needed() -> candle_core::Result<()> {
    let manual_graph = create_model();
    let mut inputs: HashMap<String, Tensor> = HashMap::new();

    inputs.insert(
        "INPUT_X".to_string(),
        Tensor::new(&[2.0f32, 3.0, 4.0], &Device::Cpu)?,
    );

    inputs.insert("MIN".to_string(), Tensor::new(&[1.0f32], &Device::Cpu)?);

    inputs.insert("MAX".to_string(), Tensor::new(&[5.0f32], &Device::Cpu)?);

    let eval = simple_eval(&manual_graph, inputs)?;

    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");
    assert_eq!(z.to_vec1::<f32>()?, vec![2.0f32, 3.0, 4.0]);

    Ok(())
}

#[test]
fn test_all_values_below_min() -> candle_core::Result<()> {
    let manual_graph = create_model();
    let mut inputs: HashMap<String, Tensor> = HashMap::new();

    inputs.insert(
        "INPUT_X".to_string(),
        Tensor::new(&[-2.0f32, -3.0, -4.0], &Device::Cpu)?,
    );

    inputs.insert("MIN".to_string(), Tensor::new(&[0.0f32], &Device::Cpu)?);

    inputs.insert("MAX".to_string(), Tensor::new(&[5.0f32], &Device::Cpu)?);

    let eval = simple_eval(&manual_graph, inputs)?;

    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");
    assert_eq!(z.to_vec1::<f32>()?, vec![0.0f32, 0.0, 0.0]);

    Ok(())
}

#[test]
fn test_all_values_below_max() -> candle_core::Result<()> {
    let manual_graph = create_model();
    let mut inputs: HashMap<String, Tensor> = HashMap::new();

    inputs.insert(
        "INPUT_X".to_string(),
        Tensor::new(&[6.0f32, 7.0, 8.0], &Device::Cpu)?,
    );

    inputs.insert("MIN".to_string(), Tensor::new(&[0.0f32], &Device::Cpu)?);

    inputs.insert("MAX".to_string(), Tensor::new(&[5.0f32], &Device::Cpu)?);

    let eval = simple_eval(&manual_graph, inputs)?;

    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");
    assert_eq!(z.to_vec1::<f32>()?, vec![5.0f32, 5.0, 5.0]);

    Ok(())
}

#[test]
fn test_clip_max_not_provided() -> candle_core::Result<()> {
    //see https://onnx.ai/onnx/operators/onnx__Clip.html
    let manual_graph = create_model();

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(
        "INPUT_X".to_string(),
        Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &Device::Cpu)?,
    );
    inputs.insert("MIN".to_string(), Tensor::new(&[2.0f32], &Device::Cpu)?);

    let eval = simple_eval(&manual_graph, inputs)?;

    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");
    assert_eq!(z.to_vec1::<f32>()?, vec![2.0f32, 2.0, 3.0, 4.0, 5.0]);

    Ok(())
}

#[test]
fn test_clip_min_not_provided() -> candle_core::Result<()> {
    //see https://onnx.ai/onnx/operators/onnx__Clip.html
    let manual_graph = create_model();

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(
        "INPUT_X".to_string(),
        Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &Device::Cpu)?,
    );
    inputs.insert("MAX".to_string(), Tensor::new(&[4.0f32], &Device::Cpu)?);

    let eval = simple_eval(&manual_graph, inputs)?;

    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");
    assert_eq!(z.to_vec1::<f32>()?, vec![1.0f32, 2.0, 3.0, 4.0, 4.0]);

    Ok(())
}

#[test]
fn test_clip_min_and_max_not_provided() -> candle_core::Result<()> {
    //see https://onnx.ai/onnx/operators/onnx__Clip.html
    let manual_graph = create_model();

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(
        "INPUT_X".to_string(),
        Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &Device::Cpu)?,
    );

    let eval = simple_eval(&manual_graph, inputs)?;

    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");
    assert_eq!(z.to_vec1::<f32>()?, vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);

    Ok(())
}

fn create_model() -> ModelProto {
    utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Clip".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec!["INPUT_X".to_string(), "MIN".to_string(), "MAX".to_string()],
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
    }))
}
