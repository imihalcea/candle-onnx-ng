use candle_core::test_utils::to_vec2_round;
use candle_core::{DType, Device, NdArray, Tensor};
use candle_onnx_ng::onnx::{GraphProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

pub mod utils;

// "Add"
#[test]
fn test_add_operation() -> candle_core::Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Add".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec!["X".to_string(), "Y".to_string()],
            output: vec!["Z".to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: "Z".to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("X".to_string(), Tensor::new(&[2.], &Device::Cpu)?);
    inputs.insert("Y".to_string(), Tensor::new(&[2.], &Device::Cpu)?);

    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("Z").expect("Output 'z' not found");
    let first = z.to_vec1::<f64>()?[0];
    assert_eq!(first, 4.0f64);
    Ok(())
}

#[test]
fn test_sub_operation() -> candle_core::Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Sub".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec!["X".to_string(), "Y".to_string()],
            output: vec!["Z".to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: "Z".to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("X".to_string(), Tensor::new(&[2.], &Device::Cpu)?);
    inputs.insert("Y".to_string(), Tensor::new(&[2.], &Device::Cpu)?);

    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("Z").expect("Output 'z' not found");
    let first = z.to_vec1::<f64>()?[0];
    assert_eq!(first, 0.0f64);
    Ok(())
}

#[test]
fn test_div_operation() -> candle_core::Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Div".to_string(),
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
    let first = z.to_vec1::<f64>()?[0];
    assert_eq!(first, 1.0f64);
    Ok(())
}

#[test]
fn test_mul_operation() -> candle_core::Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Mul".to_string(),
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
    let first = z.to_vec1::<f64>()?[0];
    assert_eq!(first, 4.0f64);
    Ok(())
}

#[test]
fn test_exp_operation() -> candle_core::Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Exp".to_string(),
            domain: "".to_string(),
            attribute: vec![],
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

    let x = Tensor::from_vec(vec![-1.0f32, 0.0f32, 1.0f32, 2.0f32], &[2, 2], &Device::Cpu)?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("INPUT_X".to_string(), x);

    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(results[0][0], 0.36787944f32);
    assert_eq!(results[0][1], 1.0f32);
    assert_eq!(results[1], vec![std::f32::consts::E, 7.389056f32]);

    Ok(())
}

#[test]
fn test_sign_operation() -> candle_core::Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Sign".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec!["X".to_string()],
            output: vec!["Z".to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: "Z".to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("X".to_string(), Tensor::arange(-2., 3., &Device::Cpu)?);

    let eval = candle_onnx_ng::simple_eval(&manual_graph, inputs)?;

    let z = eval.get("Z").expect("Output 'z' not found");
    assert_eq!(
        z.to_dtype(DType::I64)?.to_vec1::<i64>()?.to_vec(),
        vec![-1, -1, 0, 1, 1]
    );

    Ok(())
}

#[test]
fn test_pow_operation() -> candle_core::Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Pow".to_string(),
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

    let x = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32], &[2, 2], &Device::Cpu)?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("INPUT_X".to_string(), x);
    inputs.insert("INPUT_Y".to_string(), Tensor::new(2.0f32, &Device::Cpu)?);

    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(results[0], vec![1.0f32, 4.0f32]);
    assert_eq!(results[1], vec![9.0f32, 16.0f32]);

    Ok(())
}

#[test]
fn test_log() -> candle_core::Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-82
    test(&[1., 10.], &[0., std::f64::consts::LN_10])?;

    fn test(data: impl NdArray, expected: impl NdArray) -> candle_core::Result<()> {
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Log".to_string(),
                domain: "".to_string(),
                attribute: vec![],
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
        assert_eq!(eval.len(), 1);

        let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;
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

#[test]
fn test_min() -> candle_core::Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-94
    test(&[3., 2., 1.], &[1., 4., 4.], &[2., 5., 0.], &[1., 2., 0.])?;

    fn test(
        a: impl NdArray,
        b: impl NdArray,
        c: impl NdArray,
        expected: impl NdArray,
    ) -> candle_core::Result<()> {
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Min".to_string(),
                domain: "".to_string(),
                attribute: vec![],
                input: vec![
                    "INPUT_X".to_string(),
                    "INPUT_Y".to_string(),
                    "INPUT_A".to_string(),
                ],
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
        inputs.insert("INPUT_A".to_string(), Tensor::new(c, &Device::Cpu)?);

        let eval = simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;
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

// "Where"
#[test]
fn test_where() -> candle_core::Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-173
    test(
        &[[1u8, 0], [1, 1]],
        &[[1i64, 2], [3, 4]],
        &[[9i64, 8], [7, 6]],
        &[[1i64, 8], [3, 4]],
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-173
    test(
        &[[1u8, 0], [1, 1]],
        &[[1., 2.], [3., 4.]],
        &[[9., 8.], [7., 6.]],
        &[[1., 8.], [3., 4.]],
    )?;

    fn test(
        condition: impl NdArray,
        x: impl NdArray,
        y: impl NdArray,
        expected: impl NdArray,
    ) -> candle_core::Result<()> {
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Where".to_string(),
                domain: "".to_string(),
                attribute: vec![],
                input: vec![
                    "INPUT_X".to_string(),
                    "INPUT_Y".to_string(),
                    "INPUT_A".to_string(),
                ],
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
        inputs.insert("INPUT_X".to_string(), Tensor::new(condition, &Device::Cpu)?);
        inputs.insert("INPUT_Y".to_string(), Tensor::new(x, &Device::Cpu)?);
        inputs.insert("INPUT_A".to_string(), Tensor::new(y, &Device::Cpu)?);

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

// "Abs"
#[test]
fn test_abs_operation() -> candle_core::Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Abs".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec!["INPUT_X".to_string()],
            output: vec!["OUTPUT_Z".to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: "INPUT_X".to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: "INPUT_Y".to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
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
        vec![-1.0f32, 2.0f32, -3.0f32, 4.0f32],
        &[2, 2],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("INPUT_X".to_string(), x);

    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(results, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    Ok(())
}

// "Cos"
#[test]
fn test_cos_operation() -> candle_core::Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Cos".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec!["INPUT_X".to_string()],
            output: vec!["OUTPUT_Z".to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: "INPUT_X".to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: "INPUT_Y".to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
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
    assert_eq!(to_vec2_round(z, 4)?, [[1.0, 0.5403], [-0.4161, -0.99]]);
    Ok(())
}
