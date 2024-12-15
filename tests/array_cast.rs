use candle_onnx_ng::onnx::{AttributeProto, GraphProto, ModelProto, NodeProto, ValueInfoProto};

mod utils;

//TO DO: cover all tests cases according to https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast

#[test]
fn test_cast_from_f32_u8_operation() -> candle_core::Result<()> {
    let to_uint8 = 2;
    let (_, manual_graph) = create_graph_with_cast_node(to_uint8);
    let x = candle_core::Tensor::new(
        vec![1.0f32, 2.0, f32::NAN, f32::NEG_INFINITY, f32::INFINITY],
        &candle_core::Device::Cpu,
    )?;

    let mut inputs: std::collections::HashMap<String, candle_core::Tensor> =
        std::collections::HashMap::new();
    inputs.insert("INPUT_X".to_string(), x);

    let eval = candle_onnx_ng::simple_eval(&manual_graph, inputs.clone())?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");

    let results = z.to_vec1::<u8>()?;

    assert_eq!(results, vec![1u8, 2u8, 0, 0, 255]);

    Ok(())
}

fn create_graph_with_cast_node(to: i64) -> (AttributeProto, ModelProto) {
    let att_to = AttributeProto {
        name: "to".to_string(),
        ref_attr_name: "to".to_string(),
        i: to,
        doc_string: "to".to_string(),
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
            op_type: "Cast".to_string(),
            domain: "".to_string(),
            attribute: vec![att_to.clone()],
            input: vec!["INPUT_X".to_string()],
            output: vec!["OUTPUT_Z".to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![ValueInfoProto {
            name: "INPUT_X".to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
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
    (att_to, manual_graph)
}
