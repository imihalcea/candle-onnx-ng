use candle_core::{Device, NdArray, Result, Tensor};
use candle_onnx_ng::onnx::{AttributeProto, GraphProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

mod utils;

// "ReduceMax"
#[test]
fn test_reduce_l2() -> Result<()> {
    //default_axes_keepdims
    test(
        &[
            [[1., 2.], [3., 4.]],
            [[5., 6.], [7., 8.]],
            [[9., 10.], [11., 12.]],
        ],
        None,
        1,
        None,
        &[[[25.495097567963924]]],
    )?;

    //do_not_keepdims
    test(
        &[
            [[1., 2.], [3., 4.]],
            [[5., 6.], [7., 8.]],
            [[9., 10.], [11., 12.]],
        ],
        Some(vec![2]),
        0,
        None,
        &[
            [2.23606797749979, 5.0],
            [7.810249675906654, 10.63014581273465],
            [13.45362404707371, 16.278820596099706],
        ],
    )?;

    //keep dims
    test(
        &[
            [[1., 2.], [3., 4.]],
            [[5., 6.], [7., 8.]],
            [[9., 10.], [11., 12.]],
        ],
        Some(vec![2]),
        1,
        None,
        &[
            [[2.23606797749979], [5.0]],
            [[7.810249675906654], [10.63014581273465]],
            [[13.45362404707371], [16.278820596099706]],
        ],
    )?;

    //negative axes keep dims
    //TODO: not supported in current implementation
    // test(
    //     &[
    //         [[1., 2.], [3., 4.]],
    //         [[5., 6.], [7., 8.]],
    //         [[9., 10.], [11., 12.]],
    //     ],
    //     Some(vec![-1]),
    //     1,
    //     None,
    //     &[
    //         [[2.23606797749979], [5.0]],
    //         [[7.810249675906654], [10.63014581273465]],
    //         [[13.45362404707371], [16.278820596099706]]
    //     ],
    // )?;

    //noop_with_empty_axes:True, and empty axes
    test(
        &[
            [[1., 2.], [3., 4.]],
            [[5., 6.], [7., 8.]],
            [[9., 10.], [11., 12.]],
        ],
        None,
        1,
        Some(1),
        &[
            [[1., 2.], [3., 4.]],
            [[5., 6.], [7., 8.]],
            [[9., 10.], [11., 12.]],
        ],
    )?;

    fn test(
        data: impl NdArray,
        axes: Option<Vec<i64>>,
        keepdims: i64,
        noop_with_empty_axes: Option<i64>,
        expected: impl NdArray,
    ) -> Result<()> {
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

        let mut attribute = vec![att_keepdims];

        let noop = if let Some(noop_value) = noop_with_empty_axes {
            noop_value
        } else {
            0
        };

        let att_no_op_empty_axes = AttributeProto {
            name: "noop_with_empty_axes".to_string(),
            ref_attr_name: "noop_with_empty_axes".to_string(),
            i: noop,
            doc_string: "noop_with_empty_axes".to_string(),
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

        attribute.push(att_no_op_empty_axes);

        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "ReduceL2".to_string(),
                domain: "".to_string(),
                attribute,
                input: vec!["DATA".to_string(), "AXES".to_string()],
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
        let data_tensor = Tensor::new(data, &Device::Cpu)?;
        inputs.insert("DATA".to_string(), data_tensor);
        if let Some(axes_vec) = axes {
            let axes_tensor = Tensor::new(axes_vec, &Device::Cpu)?;
            inputs.insert("AXES".to_string(), axes_tensor);
        }

        let eval = simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;

        match expected.dims().len() {
            0 => {
                assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?)
            }
            1 => {
                assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?)
            }
            2 => {
                assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?)
            }
            3 => {
                assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?)
            }
            _ => unreachable!(),
        };

        Ok(())
    }
    Ok(())
}
