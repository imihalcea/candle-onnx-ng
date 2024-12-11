use candle_core::{Device, NdArray, Result, Tensor};
use candle_onnx_ng::onnx::{AttributeProto, GraphProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

mod utils;

// "ReduceMean"
#[test]
fn test_reduce_mean() -> Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-120 default_axes_keepdims
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        None,
        1,
        &[[[18.25]]],
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-120 do_no_keepdims
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![1]),
        0,
        &[[12.5, 1.5], [35.0, 1.5], [57.5, 1.5]],
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-120 keepdims
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![1]),
        1,
        &[[[12.5, 1.5]], [[35.0, 1.5]], [[57.5, 1.5]]],
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-120 negative_axes_keepdims
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![-2]),
        1,
        &[[[12.5, 1.5]], [[35.0, 1.5]], [[57.5, 1.5]]],
    )?;

    // All the test data below was generated based on numpy's np.mean
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![1, 2]),
        0,
        &[7.0, 18.25, 29.5],
    )?;

    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![1, 2]),
        1,
        &[[[7.0]], [[18.25]], [[29.5]]],
    )?;

    test(&[1., 2., 3.], None, 1, &[2.0])?;

    fn test(
        data: impl NdArray,
        axes: Option<Vec<i64>>,
        keepdims: i64,
        expected: impl NdArray,
    ) -> Result<()> {
        let has_axes = axes.is_some();

        let att_axes = AttributeProto {
            name: "axes".to_string(),
            ref_attr_name: "axes".to_string(),
            i: 0,
            doc_string: "axes".to_string(),
            r#type: 7,
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: axes.unwrap_or_default(),
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };

        let att_keepdims = AttributeProto {
            name: "keepdims".to_string(),
            ref_attr_name: "keepdims".to_string(),
            i: keepdims,
            doc_string: "keepdims".to_string(),
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
                op_type: "ReduceMean".to_string(),
                domain: "".to_string(),
                attribute: if has_axes {
                    vec![att_axes, att_keepdims]
                } else {
                    vec![att_keepdims]
                },
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
