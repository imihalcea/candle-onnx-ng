use candle_core::{Device, Tensor};
use candle_onnx_ng::onnx::{AttributeProto, GraphProto, ModelProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

mod utils;

// "Concat"
#[test]
fn test_concat_1d_operation_axis0() -> candle_core::Result<()> {
    let (mut att_axis, manual_graph) = create_graph_with_concat_node();
    let x = Tensor::new(vec![1.0f32, 2.0], &Device::Cpu)?;

    let y = Tensor::new(vec![3.0f32, 4.0], &Device::Cpu)?;
    att_axis.i = 0;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("INPUT_X".to_string(), x);
    inputs.insert("INPUT_Y".to_string(), y);

    let eval = simple_eval(&manual_graph, inputs.clone())?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");

    let results = z.to_vec1::<f32>()?;

    assert_eq!(results, vec![1.0, 2.0, 3.0, 4.0]);

    Ok(())
}

#[test]
fn test_concat_2d_operation_axis0() -> candle_core::Result<()> {
    let (mut att_axis, manual_graph) = create_graph_with_concat_node();
    let x = Tensor::new(vec![vec![1.0f32, 2.0], vec![3.0f32, 4.0]], &Device::Cpu)?;

    let y = Tensor::new(vec![vec![5.0f32, 6.0], vec![7.0f32, 8.0]], &Device::Cpu)?;
    att_axis.i = 0;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("INPUT_X".to_string(), x);
    inputs.insert("INPUT_Y".to_string(), y);

    let eval = simple_eval(&manual_graph, inputs.clone())?;
    assert_eq!(eval.len(), 1);

    let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(
        results,
        vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0]
        ]
    );

    Ok(())
}
fn create_graph_with_concat_node() -> (AttributeProto, ModelProto) {
    let att_axis = AttributeProto {
        name: "axis".to_string(),
        ref_attr_name: "axis".to_string(),
        i: 0,
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
            op_type: "Concat".to_string(),
            domain: "".to_string(),
            attribute: vec![att_axis.clone()],
            input: vec!["INPUT_X".to_string(), "INPUT_Y".to_string()],
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
    (att_axis, manual_graph)
}
