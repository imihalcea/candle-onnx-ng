use candle_core::{DType, Device, NdArray, Tensor};
use candle_onnx_ng::onnx::{GraphProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

pub mod utils;
#[test]
fn test_equal_operation() -> candle_core::Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Equal".to_string(),
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
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("INPUT_X".to_string(), Tensor::new(&[2.], &Device::Cpu)?);
    inputs.insert("INPUT_Y".to_string(), Tensor::new(&[2.], &Device::Cpu)?);

    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");
    let first = z.to_dtype(DType::U8)?.to_vec1::<u8>()?.to_vec()[0];
    assert_eq!(first, 1);

    Ok(())
}

// "Greater"
#[test]
fn test_greater() -> candle_core::Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-63
    test(&[1., 2., 3.], &[3., 2., 1.], &[0u8, 0, 1])?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-63
    test(&[1., 2., 3.], 2., &[0u8, 0, 1])?;

    fn test(a: impl NdArray, b: impl NdArray, expected: impl NdArray) -> candle_core::Result<()> {
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Greater".to_string(),
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
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert("INPUT_X".to_string(), Tensor::new(a, &Device::Cpu)?);
        inputs.insert("INPUT_Y".to_string(), Tensor::new(b, &Device::Cpu)?);

        let eval = simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval
            .get("OUTPUT_Z")
            .expect("Output 'z' not found")
            .to_dtype(DType::F64)?;

        let expected = Tensor::new(expected, &Device::Cpu)?.to_dtype(DType::F64)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}

// "Less"
#[test]
fn test_less() -> candle_core::Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-81
    test(&[1., 2., 3.], &[3., 2., 1.], &[1u8, 0, 0])?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-81
    test(&[1., 2., 3.], 2., &[1u8, 0, 0])?;

    fn test(a: impl NdArray, b: impl NdArray, expected: impl NdArray) -> candle_core::Result<()> {
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Less".to_string(),
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
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert("INPUT_X".to_string(), Tensor::new(a, &Device::Cpu)?);
        inputs.insert("INPUT_Y".to_string(), Tensor::new(b, &Device::Cpu)?);

        let eval = simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval
            .get("OUTPUT_Z")
            .expect("Output 'z' not found")
            .to_dtype(DType::F64)?;

        let expected = Tensor::new(expected, &Device::Cpu)?.to_dtype(DType::F64)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}
