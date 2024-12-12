use candle_core::{Device, Tensor};
use candle_onnx_ng::onnx::attribute_proto::AttributeType;
use candle_onnx_ng::onnx::{AttributeProto, GraphProto, ModelProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

mod utils;

fn make_graph_helper(
    op_name: &str,
    inputs: &[&str],
    outputs: &[&str],
    attribs: Vec<AttributeProto>,
) -> ModelProto {
    utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: op_name.to_string(),
            domain: "".to_string(),
            attribute: attribs,
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        input: inputs
            .iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                ..ValueInfoProto::default()
            })
            .collect(),
        output: outputs
            .iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                ..ValueInfoProto::default()
            })
            .collect(),
        ..GraphProto::default()
    }))
}

fn make_reduce_sum_graph_helper(
    inputs: &[&str],
    outputs: &[&str],
    keepdims: Option<i64>,
    noop_with_empty_axes: Option<i64>,
) -> ModelProto {
    let mut attribs = vec![];
    if let Some(keepdims) = keepdims {
        attribs.push(AttributeProto {
            name: "keepdims".to_string(),
            r#type: AttributeType::Int.into(),
            i: keepdims,
            ..AttributeProto::default()
        });
    }
    if let Some(noop_with_empty_axes) = noop_with_empty_axes {
        attribs.push(AttributeProto {
            name: "noop_with_empty_axes".to_string(),
            r#type: AttributeType::Ints.into(),
            i: noop_with_empty_axes,
            ..AttributeProto::default()
        });
    }
    make_graph_helper("ReduceSum", inputs, outputs, attribs)
}

#[test]
fn test_reduce_sum_default_axes_keepdims() -> candle_core::Result<()> {
    let manual_graph = make_reduce_sum_graph_helper(&["data", "axes"], &["reduced"], Some(1), None);

    // Test with example data
    {
        let data = Tensor::from_vec(
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            (3, 2, 2),
            &Device::Cpu,
        )?;
        // let axes = Tensor::from_vec(Vec::<i64>::new(), (0,), &Device::Cpu)?;

        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data);
        // inputs.insert("axes".to_string(), axes);

        let eval = simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let reduced = eval.get("reduced").expect("Output 'reduced' not found");
        let expected = Tensor::from_vec(vec![78.0f32], (1, 1, 1), &Device::Cpu)?;

        assert_eq!(reduced.to_vec3::<f32>()?, expected.to_vec3::<f32>()?);
    }

    {
        let data = Tensor::from_vec(
            vec![
                -5.2f32, 7.8, -3.1, 9.4, 2.6, -8.7, 4.3, -1.9, 6.5, -0.8, -7.2, 3.6,
            ],
            (3, 2, 2),
            &Device::Cpu,
        )?;

        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data.clone());

        let eval = simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let reduced = eval.get("reduced").expect("Output 'reduced' not found");
        let expected = data.sum_all()?.reshape((1, 1, 1))?;

        assert_eq!(reduced.to_vec3::<f32>()?, expected.to_vec3::<f32>()?);
    }

    Ok(())
}

#[test]
fn test_reduce_sum_do_not_keep_dims() -> candle_core::Result<()> {
    let manual_graph = make_reduce_sum_graph_helper(&["data", "axes"], &["reduced"], Some(0), None);

    // Test with example data
    {
        let data = Tensor::from_vec(
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            (3, 2, 2),
            &Device::Cpu,
        )?;
        let axes = Tensor::from_vec(vec![1i64], (1,), &Device::Cpu)?;

        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data);
        inputs.insert("axes".to_string(), axes);

        let eval = simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let reduced = eval.get("reduced").expect("Output 'reduced' not found");
        let expected = Tensor::from_vec(
            vec![4.0f32, 6.0, 12.0, 14.0, 20.0, 22.0],
            (3, 2),
            &Device::Cpu,
        )?;

        assert_eq!(reduced.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
    }

    // Test with random data
    {
        let _shape = (3, 2, 2);
        let data = Tensor::from_vec(
            vec![
                -5.2f32, 7.8, -3.1, 9.4, 2.6, -8.7, 4.3, -1.9, 6.5, -0.8, -7.2, 3.6,
            ],
            (3, 2, 2),
            &Device::Cpu,
        )?;
        let axes = Tensor::from_vec(vec![1i64], (1,), &Device::Cpu)?;

        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data.clone());
        inputs.insert("axes".to_string(), axes);

        let eval = simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let reduced = eval.get("reduced").expect("Output 'reduced' not found");

        // Calculate expected result
        let expected = data.sum(1)?;

        assert_eq!(reduced.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
    }

    Ok(())
}
