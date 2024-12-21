use candle_core::{Device, Tensor};
use candle_onnx_ng::onnx::attribute_proto::AttributeType;
use candle_onnx_ng::onnx::{AttributeProto, GraphProto, ModelProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

pub mod utils;

// "MatMul"
#[test]
fn gemm_with_all_attributes() -> candle_core::Result<()> {
    //https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm
    let manual_graph = create_gemm_graph(Some(0.25), Some(0.35), Some(1), Some(1));

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(
        "A".to_string(),
        Tensor::from_vec(
            //
            vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
            &[2, 2],
            &Device::Cpu,
        )?,
    );
    inputs.insert(
        "B".to_string(),
        Tensor::from_vec(
            //
            vec![5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32, 10.0f32],
            &[2, 3],
            &Device::Cpu,
        )?,
    );

    inputs.insert(
        "C".to_string(),
        Tensor::from_vec(
            //
            vec![1.0f32, 2.0f32, 3.0f32],
            &[1, 3],
            &Device::Cpu,
        )?,
    );
    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");
    let results = z.to_vec2::<f32>()?;
    assert_eq!(results, vec![vec![5.6, 6.7, 7.8], vec![12.1, 14.2, 16.3]]);

    Ok(())
}

#[test]
fn gemm_with_alpha() -> candle_core::Result<()> {
    let manual_graph = create_gemm_graph(Some(0.25), None, None, None);

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(
        "A".to_string(),
        Tensor::from_vec(
            //
            vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
            &[2, 2],
            &Device::Cpu,
        )?,
    );
    inputs.insert(
        "B".to_string(),
        Tensor::from_vec(
            //
            vec![5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32, 10.0f32],
            &[2, 3],
            &Device::Cpu,
        )?,
    );

    inputs.insert(
        "C".to_string(),
        Tensor::from_vec(
            //
            vec![0f32, 0f32, 0f32],
            &[1, 3],
            &Device::Cpu,
        )?,
    );
    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");
    let results = z.to_vec2::<f32>()?;
    assert_eq!(
        results,
        vec![vec![5.25, 6.0, 6.75], vec![11.75, 13.5, 15.25]]
    );

    Ok(())
}

#[test]
fn gemm_with_beta() -> candle_core::Result<()> {
    let manual_graph = create_gemm_graph(None, Some(0.35), None, None);

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(
        "A".to_string(),
        Tensor::from_vec(
            //
            vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
            &[2, 2],
            &Device::Cpu,
        )?,
    );
    inputs.insert(
        "B".to_string(),
        Tensor::from_vec(
            //
            vec![5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32, 10.0f32],
            &[2, 3],
            &Device::Cpu,
        )?,
    );

    inputs.insert(
        "C".to_string(),
        Tensor::from_vec(
            //
            vec![0f32, 0f32, 0f32],
            &[1, 3],
            &Device::Cpu,
        )?,
    );
    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");
    let results = z.to_vec2::<f32>()?;
    assert_eq!(
        results,
        vec![vec![21.0, 24.0, 27.0], vec![47.0, 54.0, 61.0]]
    );

    Ok(())
}

#[test]
#[ignore] //see fixme in gemm.rs
fn gemm_no_bias() -> candle_core::Result<()> {
    let manual_graph = create_gemm_graph(None, None, None, None);

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(
        "A".to_string(),
        Tensor::from_vec(
            //
            vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
            &[2, 2],
            &Device::Cpu,
        )?,
    );
    inputs.insert(
        "B".to_string(),
        Tensor::from_vec(
            //
            vec![5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32, 10.0f32],
            &[2, 3],
            &Device::Cpu,
        )?,
    );

    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");
    let results = z.to_vec2::<f32>()?;
    assert_eq!(results, vec![vec![5.0, 6.0], vec![11.0, 13.0]]);

    Ok(())
}

fn create_gemm_graph(
    alpha_value: Option<f32>,
    beta_value: Option<f32>,
    t_a: Option<i64>,
    t_b: Option<i64>,
) -> ModelProto {
    let alpha = AttributeProto {
        name: "alpha".to_string(),
        ref_attr_name: "alpha".to_string(),
        i: 0,
        doc_string: "alpha".to_string(),
        r#type: AttributeType::Float.into(),
        f: alpha_value.unwrap_or(1.0),
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

    let beta = AttributeProto {
        name: "beta".to_string(),
        ref_attr_name: "beta".to_string(),
        i: 0,
        doc_string: "beta".to_string(),
        r#type: AttributeType::Float.into(),
        f: beta_value.unwrap_or(1.0),
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

    let trans_a = AttributeProto {
        name: "trans_a".to_string(),
        ref_attr_name: "trans_a".to_string(),
        i: t_a.unwrap_or(0),
        doc_string: "trans_a".to_string(),
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

    let trans_b = AttributeProto {
        name: "trans_b".to_string(),
        ref_attr_name: "trans_b".to_string(),
        i: t_b.unwrap_or(0),
        doc_string: "trans_b".to_string(),
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

    let mut attributes: Vec<AttributeProto> = vec![];
    if alpha_value.is_some() {
        attributes.push(alpha);
    }
    if beta_value.is_some() {
        attributes.push(beta);
    }
    if t_a.is_some() {
        attributes.push(trans_a);
    }
    if t_b.is_some() {
        attributes.push(trans_b);
    }

    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Gemm".to_string(),
            domain: "".to_string(),
            attribute: attributes,
            input: vec!["A".to_string(), "B".to_string(), "C".to_string()],
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
