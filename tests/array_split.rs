use candle_core::{Device, Tensor};
use candle_onnx_ng::onnx::attribute_proto::AttributeType;
use candle_onnx_ng::onnx::{AttributeProto, GraphProto, ModelProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

mod utils;
fn make_split_graph_helper(inputs: &[&str], outputs: &[&str], axis: i64) -> ModelProto {
    let attribs = vec![AttributeProto {
        name: "axis".to_string(),
        r#type: AttributeType::Int.into(),
        i: axis,
        ..AttributeProto::default()
    }];

    make_graph_helper("Split", inputs, outputs, attribs)
}

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

#[test]
fn test_split_equal_parts_1d_opset13() -> candle_core::Result<()> {
    let input = Tensor::from_vec(
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32],
        (6,),
        &Device::Cpu,
    )?;
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), input);

    {
        let manual_graph =
            make_split_graph_helper(&["input"], &["output_1", "output_2", "output_3"], 0);
        let eval = simple_eval(&manual_graph, inputs.clone())?;
        assert_eq!(eval.len(), 3);

        let out1 = eval.get("output_1").expect("Output 'output_1' not found");
        let out2 = eval.get("output_2").expect("Output 'output_2' not found");
        let out3 = eval.get("output_3").expect("Output 'output_3' not found");

        assert_eq!(out1.to_vec1::<f32>()?, vec![1.0f32, 2.0f32]);
        assert_eq!(out2.to_vec1::<f32>()?, vec![3.0f32, 4.0f32]);
        assert_eq!(out3.to_vec1::<f32>()?, vec![5.0f32, 6.0f32]);
    }

    {
        let splits = Tensor::from_vec(vec![2i64, 4], (2,), &Device::Cpu)?;
        inputs.insert("split".to_string(), splits);

        let manual_graph =
            make_split_graph_helper(&["input", "split"], &["output_1", "output_2"], 0);
        let eval = simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 2);

        let out1 = eval.get("output_1").expect("Output 'output_1' not found");
        let out2 = eval.get("output_2").expect("Output 'output_2' not found");

        assert_eq!(out1.to_vec1::<f32>()?, vec![1.0f32, 2.0f32]);
        assert_eq!(out2.to_vec1::<f32>()?, vec![3.0f32, 4.0f32, 5.0f32, 6.0f32]);
    }
    Ok(())
}
