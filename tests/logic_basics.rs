use candle_core::{DType, Device, NdArray, Result, Tensor};
use std::collections::HashMap;
use candle_onnx_ng::onnx::{GraphProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;

pub mod utils;

// "Not"
#[test]
fn test_not_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Not".to_string(),
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
    inputs.insert("INPUT_X".to_string(), Tensor::new(&[0.], &Device::Cpu)?);

    let eval = simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");
    let first = z.to_dtype(DType::U8)?.to_vec1::<u8>()?.to_vec()[0];
    assert_eq!(first, 1);

    Ok(())
}


// Xor
#[test]
fn test_xor() -> Result<()> {
    // tests based on: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Xor xor

    // 2d
    test(
        &[[0_u8, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 1]],
        &[[1_u8, 1, 0, 0], [1, 0, 0, 1], [1, 1, 1, 0]],
        &[[1_u8, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]],
    )?;

    // 3d
    test(
        &[
            [
                [0_u8, 1, 1, 1, 1],
                [0, 1, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1],
            ],
            [
                [0, 0, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 1, 0, 0, 1],
                [1, 0, 0, 1, 0],
            ],
            [
                [1, 0, 0, 1, 1],
                [1, 1, 1, 0, 0],
                [1, 1, 0, 0, 1],
                [1, 0, 0, 0, 1],
            ],
        ],
        &[
            [
                [1_u8, 0, 0, 1, 1],
                [0, 0, 1, 0, 1],
                [1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [1, 0, 0, 1, 1],
                [1, 0, 1, 1, 1],
                [0, 1, 0, 1, 1],
                [1, 1, 1, 0, 0],
            ],
            [
                [0, 1, 1, 1, 0],
                [1, 1, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 0, 1, 0],
            ],
        ],
        &[
            [
                [1_u8, 1, 1, 0, 0],
                [0, 1, 0, 0, 1],
                [0, 1, 1, 0, 1],
                [0, 0, 0, 0, 1],
            ],
            [
                [1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 1, 0],
                [0, 1, 1, 1, 0],
            ],
            [
                [1, 1, 1, 0, 1],
                [0, 0, 1, 1, 0],
                [1, 0, 1, 1, 1],
                [0, 1, 0, 1, 1],
            ],
        ],
    )?;

    // 4d
    test(
        &[
            [
                [[0_u8, 1, 1, 0], [1, 0, 0, 0], [1, 1, 0, 1]],
                [[1, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
            ],
            [
                [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]],
                [[1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1]],
            ],
        ],
        &[
            [
                [[1_u8, 0, 1, 0], [0, 0, 1, 1], [1, 0, 1, 0]],
                [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
            ],
            [
                [[1, 1, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]],
            ],
        ],
        &[
            [
                [[1_u8, 1, 0, 0], [1, 0, 1, 1], [0, 1, 1, 1]],
                [[1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 0, 0]],
            ],
            [
                [[0, 0, 1, 0], [1, 0, 1, 1], [1, 0, 1, 0]],
                [[1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0]],
            ],
        ],
    )?;

    // tests based on: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Xor xor_broadcast
    // 3d vs 1d
    test(
        // Shape (3, 4, 5)
        &[
            [
                [0_u8, 0, 0, 0, 1],
                [0, 1, 0, 1, 1],
                [1, 0, 0, 1, 1],
                [0, 0, 1, 0, 1],
            ],
            [
                [0, 1, 0, 1, 1],
                [1, 1, 0, 0, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            [
                [1, 1, 0, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 1, 1, 0, 1],
                [1, 1, 0, 1, 1],
            ],
        ],
        // shape (5)
        &[1_u8, 0, 0, 1, 1],
        // shape (3, 4, 5)
        &[
            [
                [1_u8, 0, 0, 1, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 1, 1, 0],
            ],
            [
                [1, 1, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [1, 1, 1, 0, 1],
                [1, 0, 0, 1, 0],
            ],
            [
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0],
                [0, 1, 0, 0, 0],
            ],
        ],
    )?;

    // 3d vs 2d
    test(
        // Shape (3, 4, 5)
        &[
            [
                [0_u8, 0, 0, 0, 1],
                [0, 1, 0, 1, 1],
                [1, 0, 0, 1, 1],
                [0, 0, 1, 0, 1],
            ],
            [
                [0, 1, 0, 1, 1],
                [1, 1, 0, 0, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            [
                [1, 1, 0, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 1, 1, 0, 1],
                [1, 1, 0, 1, 1],
            ],
        ],
        // shape (4, 5)
        &[
            [0_u8, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 0],
        ],
        // shape (3, 4, 5)
        &[
            [
                [0_u8, 1, 0, 1, 1],
                [0, 1, 1, 1, 1],
                [0, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ],
            [
                [0, 0, 0, 0, 1],
                [1, 1, 1, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 0, 1, 1],
            ],
            [
                [1, 0, 0, 0, 1],
                [0, 0, 1, 1, 1],
                [1, 0, 1, 1, 0],
                [0, 0, 0, 0, 1],
            ],
        ],
    )?;

    // 4d vs 2d
    test(
        // Shape (2, 3, 3, 4)
        &[
            [
                [[1_u8, 0, 0, 1], [1, 1, 0, 0], [0, 1, 0, 0]],
                [[1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 1]],
                [[1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]],
            ],
            [
                [[0, 1, 0, 1], [1, 1, 0, 1], [1, 0, 1, 1]],
                [[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1]],
                [[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 1]],
            ],
        ],
        // shape (3, 4)
        &[[0_u8, 0, 1, 1], [1, 1, 1, 1], [0, 1, 0, 1]],
        // shape (2, 3, 3, 4)
        &[
            [
                [[1_u8, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
                [[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 0]],
                [[1, 0, 1, 1], [0, 0, 0, 1], [0, 1, 1, 0]],
            ],
            [
                [[0, 1, 1, 0], [0, 0, 1, 0], [1, 1, 1, 0]],
                [[1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 0]],
                [[1, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]],
            ],
        ],
    )?;

    // 4d vs 3d
    test(
        // Shape (2, 3, 3, 4)
        &[
            [
                [[1_u8, 0, 0, 1], [1, 1, 0, 0], [0, 1, 0, 0]],
                [[1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 1]],
                [[1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]],
            ],
            [
                [[0, 1, 0, 1], [1, 1, 0, 1], [1, 0, 1, 1]],
                [[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1]],
                [[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 1]],
            ],
        ],
        // shape (3, 3, 4)
        &[
            [[1_u8, 1, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0]],
            [[0, 1, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1]],
            [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1]],
        ],
        // shape (2, 3, 3, 4)
        &[
            [
                [[0_u8, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0]],
                [[1, 0, 0, 1], [0, 1, 0, 0], [1, 1, 0, 0]],
                [[1, 1, 1, 0], [0, 1, 0, 1], [1, 1, 1, 0]],
            ],
            [
                [[1, 0, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]],
                [[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 1, 0]],
                [[1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0]],
            ],
        ],
    )?;

    // 4d vs 4d
    test(
        // Shape (1, 4, 1, 2)
        &[[[[1_u8, 0]], [[1, 0]], [[1, 0]], [[1, 1]]]],
        // shape (2, 1, 4, 2)
        &[
            [[[0_u8, 0], [1, 1], [1, 1], [1, 1]]],
            [[[0, 1], [1, 0], [0, 1], [0, 0]]],
        ],
        // shape (2, 4, 4, 2)
        &[
            [
                [[1_u8, 0], [0, 1], [0, 1], [0, 1]],
                [[1, 0], [0, 1], [0, 1], [0, 1]],
                [[1, 0], [0, 1], [0, 1], [0, 1]],
                [[1, 1], [0, 0], [0, 0], [0, 0]],
            ],
            [
                [[1, 1], [0, 0], [1, 1], [1, 0]],
                [[1, 1], [0, 0], [1, 1], [1, 0]],
                [[1, 1], [0, 0], [1, 1], [1, 0]],
                [[1, 0], [0, 1], [1, 0], [1, 1]],
            ],
        ],
    )?;

    fn test(input: impl NdArray, other: impl NdArray, expected: impl NdArray) -> Result<()> {
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Xor".to_string(),
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

        let inputs: HashMap<String, Tensor> = HashMap::from([
            ("X".to_string(), Tensor::new(input, &Device::Cpu)?),
            ("Y".to_string(), Tensor::new(other, &Device::Cpu)?),
        ]);

        let eval = candle_onnx_ng::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval.get("Z").expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;

        match expected.dims().len() {
            0 => {
                assert_eq!(z.to_vec0::<u8>()?, expected.to_vec0::<u8>()?)
            }
            1 => {
                assert_eq!(z.to_vec1::<u8>()?, expected.to_vec1::<u8>()?)
            }
            2 => {
                assert_eq!(z.to_vec2::<u8>()?, expected.to_vec2::<u8>()?)
            }
            3 => {
                assert_eq!(z.to_vec3::<u8>()?, expected.to_vec3::<u8>()?)
            }
            4 => {
                // Candle has no method equivallent to `to_vec4()`
                // So, as a hack, we flatten it to a single dim vec to test the results
                assert_eq!(
                    z.flatten_all()?.to_vec1::<u8>()?,
                    expected.flatten_all()?.to_vec1::<u8>()?
                )
            }
            _ => unreachable!(),
        };

        Ok(())
    }
    Ok(())
}