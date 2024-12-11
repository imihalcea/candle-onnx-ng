use std::collections::HashMap;
use candle_core::{DType, Device, NdArray, Tensor, Result};
use candle_onnx_ng::onnx::{AttributeProto, GraphProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;

mod utils;
// "ReduceMin"
#[test]
fn test_reduce_min() -> candle_core::Result<()> {
    // Tests with random data generated with `np.random.uniform`
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-121 bool_inputs
    // No special treatment reqired for bool
    // `np.minimum.reduce(data, axis=axes, keepdims=True)`
    test(
        &[[1_u8, 1], [1, 0], [0, 1], [0, 0]],
        Some(vec![1]),
        1,
        None,
        &[[1_u8], [0], [0], [0]],
        false,
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-121 default_axes_keepdims
    // `np.minimum.reduce(data, axis=None, keepdims=True)`
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        None,
        1,
        None,
        &[[[1.]]],
        false,
    )?;
    // same as above but with random
    test(
        &[
            [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
            [[6.304022, 4.939862], [4.5435624, 3.072864]],
            [[-2.5058026, 8.008944], [9.587318, -8.794852]],
        ],
        None,
        1,
        None,
        &[[[-8.794852]]],
        false,
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-121 default_axes_donot_keep_dims
    // `np.minimum.reduce(data, axis=None, keepdims=False)`
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        None,
        0,
        None,
        1.,
        false,
    )?;
    // same as above but with random
    // `np.minimum.reduce(data, axis=None, keepdims=False)`
    test(
        &[
            [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
            [[6.304022, 4.939862], [4.5435624, 3.072864]],
            [[-2.5058026, 8.008944], [9.587318, -8.794852]],
        ],
        None,
        0,
        None,
        -8.794852,
        false,
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-121 keepdims
    // `np.minimum.reduce(data, axis=tuple(axes), keepdims=True)`
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![1]),
        1,
        None,
        &[[[5., 1.]], [[30., 1.]], [[55., 1.]]],
        false,
    )?;
    // keepdims with random data
    // `np.minimum.reduce(data, axis=tuple(axes), keepdims=True)`
    test(
        &[
            [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
            [[6.304022, 4.939862], [4.5435624, 3.072864]],
            [[-2.5058026, 8.008944], [9.587318, -8.794852]],
        ],
        Some(vec![1]),
        1,
        None,
        &[
            [[-7.648377, -5.4018507]],
            [[4.5435624, 3.072864]],
            [[-2.5058026, -8.794852]],
        ],
        false,
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-121 negative_axes_keepdims
    // axes = np.array([-1], dtype=np.int64)
    // `np.minimum.reduce(data, axis=tuple(axes), keepdims=True)`
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![-1]),
        1,
        None,
        &[[[1.], [2.]], [[1.], [2.]], [[1.], [2.]]],
        false,
    )?;
    // axes = np.array([-2], dtype=np.int64)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![-2]),
        1,
        None,
        &[[[5., 1.]], [[30., 1.]], [[55., 1.]]],
        false,
    )?;
    // with random
    test(
        &[
            [[-4.1676497, -2.7603748], [-4.5138783, -0.762791]],
            [[-6.3792877, 7.1619177], [-9.958144, 6.3753467]],
            [[9.046973, 3.4554052], [-5.4674335, 5.4642754]],
        ],
        Some(vec![-2]),
        1,
        None,
        &[
            [[-4.5138783, -2.7603748]],
            [[-9.958144, 6.3753467]],
            [[-5.4674335, 3.4554052]],
        ],
        false,
    )?;

    // Multiple axes - keepdims=1 (true)
    // axes = np.array([0, 1], dtype=np.int64)
    // np.minimum.reduce(data, axis=tuple(axes), keepdims=True)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![0, 1]),
        1,
        None,
        &[[[5., 1.]]],
        false,
    )?;
    // axes = np.array([0, 2], dtype=np.int64)
    // np.minimum.reduce(data, axis=tuple(axes), keepdims=True)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![0, 2]),
        1,
        None,
        &[[[1.], [2.]]],
        false,
    )?;
    // axes = np.array([2, 1], dtype=np.int64)
    // np.minimum.reduce(data, axis=tuple(axes), keepdims=True)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![2, 1]),
        1,
        None,
        &[[[1.]], [[1.]], [[1.]]],
        false,
    )?;
    // axes = np.array([2, 0, 1], dtype=np.int64)
    // np.minimum.reduce(data, axis=tuple(axes), keepdims=True)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![2, 0, 1]),
        1,
        None,
        &[[[1.]]],
        false,
    )?;
    // Multiple axes - keepdims=0 (false)
    // axes = np.array([0, 1], dtype=np.int64)
    // np.minimum.reduce(data, axis=tuple(axes), keepdims=False)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![0, 1]),
        0,
        None,
        &[5., 1.],
        false,
    )?;
    // axes = np.array([0, 2], dtype=np.int64)
    // np.minimum.reduce(data, axis=tuple(axes), keepdims=False)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![0, 2]),
        0,
        None,
        &[1., 2.],
        false,
    )?;
    // axes = np.array([2, 1], dtype=np.int64)
    // np.minimum.reduce(data, axis=tuple(axes), keepdims=False)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![2, 1]),
        0,
        None,
        &[1., 1., 1.],
        false,
    )?;
    // axes = np.array([2, 0, 1], dtype=np.int64)
    // np.minimum.reduce(data, axis=tuple(axes), keepdims=False)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![2, 0, 1]),
        0,
        None,
        1.,
        false,
    )?;

    // Multiple axes - negative `axes` - keepdims=1 (true)
    // axes = np.array([-1, 0, 1], dtype=np.int64)
    // np.minimum.reduce(data, axis=tuple(axes), keepdims=True)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![-1, 0, 1]),
        1,
        None,
        &[[[1.]]],
        false,
    )?;
    // Multiple axes - negative `axes` - keepdims=0 (false)
    // axes = np.array([-1, 0, 1], dtype=np.int64)
    // np.minimum.reduce(data, axis=tuple(axes), keepdims=True)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![-1, 0, 1]),
        0,
        None,
        1.,
        false,
    )?;

    // `noop_with_empty_axes = true (1)` should yield tensor equivallent to the input tensor
    test(
        &[
            [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
            [[6.304022, 4.939862], [4.5435624, 3.072864]],
            [[-2.5058026, 8.008944], [9.587318, -8.794852]],
        ],
        None,
        0,
        Some(1),
        &[
            [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
            [[6.304022, 4.939862], [4.5435624, 3.072864]],
            [[-2.5058026, 8.008944], [9.587318, -8.794852]],
        ],
        false,
    )?;

    // Rank-0 tensors are also valid
    test(42., None, 0, None, 42., false)?;
    test(42., None, 1, None, 42., false)?;

    // Negative test - expect error
    // axes = np.array([-2, 0, 1], dtype=np.int64)
    // np.minimum.reduce(data, axis=tuple(axes), keepdims=True)
    // Should error out with `duplicate value in "axes"`
    assert!(test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![-2, 0, 1]),
        1,
        None,
        &[0.],
        false
    )
        .is_err());

    // Negative test - expect error
    // Should error out on empty set
    assert!(test(&[[1_u8; 0]], Some(vec![-2, 0, 1]), 1, None, &[0.], false).is_err());

    // Backward compatibility
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![-1, 0, 1]),
        0,
        None,
        1.,
        true,
    )?;

    fn test(
        data: impl NdArray,
        axes: Option<Vec<i64>>,
        keepdims: i64,
        noop_with_empty_axes: Option<i64>,
        expected: impl NdArray,
        backward_comp: bool,
    ) -> Result<()> {
        let has_axes = axes.is_some();

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
        if let Some(noop) = noop_with_empty_axes {
            if !has_axes {
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
            }
        }
        if has_axes && backward_comp {
            attribute.push(AttributeProto {
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
                ints: axes.clone().unwrap_or_default(),
                strings: vec![],
                tensors: vec![],
                graphs: vec![],
                sparse_tensors: vec![],
                type_protos: vec![],
            });
        }

        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "ReduceMin".to_string(),
                domain: "".to_string(),
                attribute,
                input: if has_axes && !backward_comp {
                    vec!["INPUT_X".to_string(), "INPUT_Y".to_string()]
                } else {
                    vec!["INPUT_X".to_string()]
                },
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
        let input_tensor = Tensor::new(data, &Device::Cpu)?;
        let input_dtype = input_tensor.dtype();
        inputs.insert("INPUT_X".to_string(), input_tensor);
        if !backward_comp {
            if let Some(a) = axes {
                inputs.insert("INPUT_Y".to_string(), Tensor::new(a, &Device::Cpu)?);
            }
        }

        let eval = simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;

        match expected.dims().len() {
            0 => {
                if input_dtype == DType::U8 {
                    assert_eq!(z.to_vec0::<u8>()?, expected.to_vec0::<u8>()?)
                } else {
                    assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?)
                }
            }
            1 => {
                if input_dtype == DType::U8 {
                    assert_eq!(z.to_vec1::<u8>()?, expected.to_vec1::<u8>()?)
                } else {
                    assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?)
                }
            }
            2 => {
                if input_dtype == DType::U8 {
                    assert_eq!(z.to_vec2::<u8>()?, expected.to_vec2::<u8>()?)
                } else {
                    assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?)
                }
            }
            3 => {
                if input_dtype == DType::U8 {
                    assert_eq!(z.to_vec3::<u8>()?, expected.to_vec3::<u8>()?)
                } else {
                    assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?)
                }
            }
            _ => unreachable!(),
        };

        Ok(())
    }
    Ok(())
}