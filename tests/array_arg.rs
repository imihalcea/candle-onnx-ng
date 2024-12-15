use std::collections::HashMap;
use candle_core::{Device, NdArray, Tensor};
use candle_onnx_ng::onnx::{AttributeProto, GraphProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;

mod utils;

// "ArgMin"
#[test]
fn test_argmin() -> candle_core::Result<()> {
    // tests from https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-7
    // default_axes_keepdims
    test(
        &[[2u32, 1u32], [3u32, 10u32]],
        None,
        Some(1),
        None,
        &[[0i64, 0i64]],
    )?;
    // keepdims
    test(
        &[[2u32, 1u32], [3u32, 10u32]],
        Some(1),
        Some(1),
        None,
        &[[1i64], [0i64]],
    )?;
    // // negative_axis_keepdims
    test(
        &[[2u32, 1u32], [3u32, 10u32]],
        Some(-1),
        Some(1),
        None,
        &[[1i64], [0i64]],
    )?;
    // no_keepdims
    test(
        &[[2u32, 1u32], [3u32, 10u32]],
        None,
        Some(0),
        None,
        &[0i64, 0i64],
    )?;
    // tests from https://pytorch.org/docs/stable/generated/torch.argmin.html#torch.argmin
    test(
        &[
            [0.1139, 0.2254, -0.1381, 0.3687],
            [1.0100, -1.1975, -0.0102, -0.4732],
            [-0.9240, 0.1207, -0.7506, -1.0213],
            [1.7809, -1.2960, 0.9384, 0.1438],
        ],
        Some(1),
        Some(0),
        None,
        &[2i64, 1i64, 3i64, 1i64],
    )?;
    test(
        &[
            [0.1139, 0.2254, -0.1381, 0.3687],
            [1.0100, -1.1975, -0.0102, -0.4732],
            [-0.9240, 0.1207, -0.7506, -1.0213],
            [1.7809, -1.2960, 0.9384, 0.1438],
        ],
        Some(1),
        None,
        None,
        &[[2i64], [1i64], [3i64], [1i64]],
    )?;
    fn test(
        data: impl NdArray,
        axis: Option<i64>,
        keepdims: Option<i64>,
        select_last_index: Option<i64>,
        expected: impl NdArray,
    ) -> candle_core::Result<()> {
        let att_axis = AttributeProto {
            name: "axis".to_string(),
            ref_attr_name: "axis".to_string(),
            i: axis.unwrap_or(0),
            doc_string: "axis".to_string(),
            r#type: 2, // INT
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
        let att_keepdims = AttributeProto {
            name: "keepdims".to_string(),
            ref_attr_name: "keepdims".to_string(),
            i: keepdims.unwrap_or(1),
            doc_string: "keepdims".to_string(),
            r#type: 2, // INT
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
        let att_select_last_index = AttributeProto {
            name: "select_last_index".to_string(),
            ref_attr_name: "select_last_index".to_string(),
            i: select_last_index.unwrap_or(0),
            doc_string: "select_last_index".to_string(),
            r#type: 2, // INT
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
        let attrs = {
            let mut mut_attrs = vec![];
            if axis.is_some() {
                mut_attrs.push(att_axis);
            }
            if keepdims.is_some() {
                mut_attrs.push(att_keepdims);
            }
            if select_last_index.is_some() {
                mut_attrs.push(att_select_last_index);
            }
            mut_attrs
        };
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "ArgMin".to_string(),
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
        match expected.dims().len() {
            1 => assert_eq!(z.to_vec1::<i64>()?, expected.to_vec1::<i64>()?),
            2 => assert_eq!(z.to_vec2::<i64>()?, expected.to_vec2::<i64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}
