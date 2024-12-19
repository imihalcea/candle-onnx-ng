use candle_core::{Device, NdArray, Tensor};
use candle_onnx_ng::onnx::{AttributeProto, GraphProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

pub mod utils;

// "Sigmoid"
#[test]
fn test_sigmoid_operation() -> candle_core::Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Sigmoid".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec!["INPUT_X".to_string()],
            output: vec!["OUTPUT_Z".to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![ValueInfoProto {
            name: "INPUT_X".to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
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
    let x = Tensor::from_vec(vec![0.0f32, 1.0f32, 2.0f32, 3.0f32], &[2, 2], &Device::Cpu)?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("INPUT_X".to_string(), x);

    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(
        results,
        vec![vec![0.5, 0.7310586], vec![0.880797, 0.95257413]]
    );

    Ok(())
}

#[test]
fn test_gelu_operation() -> candle_core::Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Gelu".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec!["INPUT_X".to_string()],
            output: vec!["OUTPUT_Z".to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![ValueInfoProto {
            name: "INPUT_X".to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
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
    let x = Tensor::from_vec(vec![0.0f32, 1.0f32, 2.0f32, 3.0f32], &[2, 2], &Device::Cpu)?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("INPUT_X".to_string(), x);

    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(
        results,
        vec![vec![0.0, 0.8413448], vec![1.9544997, 2.9959502]]
    );

    Ok(())
}

#[test]
fn test_relu_operation() -> candle_core::Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Relu".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec!["INPUT_X".to_string()],
            output: vec!["OUTPUT_Z".to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![ValueInfoProto {
            name: "INPUT_X".to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
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
        vec![-1.0f32, 1.0f32, -2.0f32, 3.0f32],
        &[2, 2],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("INPUT_X".to_string(), x);

    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(results, vec![vec![0.0, 1.0], vec![0.0, 3.0]]);

    Ok(())
}

#[test]
fn test_leakyrelu() -> candle_core::Result<()> {
    // tests from https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-80
    // leakyrelu
    test(&[-1.0, 0.0, 1.0], Some(0.1), &[-0.1, 0.0, 1.0])?;
    fn test(data: impl NdArray, alpha: Option<f32>, expected: impl NdArray) -> candle_core::Result<()> {
        let att_alpha = AttributeProto {
            name: "alpha".to_string(),
            ref_attr_name: "alpha".to_string(),
            i: 0,
            doc_string: "alpha".to_string(),
            r#type: 1, // FLOAT
            f: alpha.unwrap_or(0.01),
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
        let attrs = {
            let mut mut_attrs = vec![];
            if alpha.is_some() {
                mut_attrs.push(att_alpha);
            }
            mut_attrs
        };
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "LeakyRelu".to_string(),
                domain: "".to_string(),
                attribute: attrs,
                input: vec!["INPUT_X".to_string()],
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
        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert("INPUT_X".to_string(), Tensor::new(data, &Device::Cpu)?);
        let eval = simple_eval(&manual_graph, inputs)?;
        let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;
        for both in z
            .to_vec1::<f64>()?
            .iter()
            .zip(expected.to_vec1::<f64>()?.iter())
        {
            let (act, exp) = both;
            assert!(f64::abs(act - exp) < f32::EPSILON.into());
        }

        Ok(())
    }

    Ok(())
}