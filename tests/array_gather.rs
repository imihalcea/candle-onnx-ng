use candle_core::{Device, NdArray, Tensor};
use candle_onnx_ng::onnx::{AttributeProto, GraphProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;
pub mod utils;

// "Gather"
#[test]
fn test_gather_operation() -> candle_core::Result<()> {
    // test taken from https://onnx.ai/onnx/operators/onnx__Gather.html#summary.
    test(
        &[[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]],
        &[[0i64, 1], [1, 2]],
        0,
        &[[[1.0, 1.2], [2.3, 3.4]], [[2.3, 3.4], [4.5, 5.7]]],
    )?;

    // test taken from https://onnx.ai/onnx/operators/onnx__Gather.html#summary.
    test(
        &[[1.0, 1.2, 1.9], [2.3, 3.4, 3.9], [4.5, 5.7, 5.9]],
        &[[0i64, 2]],
        1,
        &[[[1.0, 1.9]], [[2.3, 3.9]], [[4.5, 5.9]]],
    )?;

    // all the tests below are generated from numpy.take, which works like
    // onnx's Gather operation.
    test(&[1.0, 2.0, 3.0, 4.0], 3i64, 0, 4.0)?;

    test(&[[1.0, 2.0, 3.0, 4.0]], 3i64, 1, &[4.0])?;

    test(
        &[[1.0], [2.0], [3.0], [4.0]],
        &[3i64, 2],
        0,
        &[[4.0], [3.0]],
    )?;

    test(
        &[
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
            [[13.0, 14.0], [15.0, 16.0]],
        ],
        1i64,
        0,
        &[[5.0, 6.0], [7.0, 8.0]],
    )?;

    test(
        &[
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
            [[13.0, 14.0], [15.0, 16.0]],
        ],
        &[1i64, 0],
        0,
        &[[[5.0, 6.0], [7.0, 8.0]], [[1.0, 2.0], [3.0, 4.0]]],
    )?;

    fn test(
        data: impl NdArray,
        indices: impl NdArray,
        axis: i64,
        expected: impl NdArray,
    ) -> candle_core::Result<()> {
        let att_axis = AttributeProto {
            name: "axis".to_string(),
            ref_attr_name: "axis".to_string(),
            i: axis,
            doc_string: "axis".to_string(),
            r#type: 2,
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
                op_type: "Gather".to_string(),
                domain: "".to_string(),
                attribute: vec![att_axis],
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
        inputs.insert("INPUT_X".to_string(), Tensor::new(data, &Device::Cpu)?);
        inputs.insert("INPUT_Y".to_string(), Tensor::new(indices, &Device::Cpu)?);

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
