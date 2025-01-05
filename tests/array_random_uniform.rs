use candle_core::Result;
use candle_onnx_ng::onnx::{AttributeProto, GraphProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

mod utils;

// "RandomUniform"
#[test]
fn test_random_uniform() -> Result<()> {
    test(vec![3, 2, 1, 4], None, None)?;
    test(vec![2, 2, 2, 2], Some(-10.0), None)?;
    test(vec![2, 2, 2, 2], None, Some(10.0))?;
    test(vec![1, 2, 3, 4], Some(-10.0), Some(10.0))?;

    fn test(shape: Vec<i64>, low: Option<f32>, high: Option<f32>) -> Result<()> {
        let att_low = AttributeProto {
            name: "low".to_string(),
            ref_attr_name: "low".to_string(),
            i: 0,
            doc_string: "low".to_string(),
            r#type: 1, // FLOAT
            f: low.unwrap_or(0.0),
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
        let att_high = AttributeProto {
            name: "high".to_string(),
            ref_attr_name: "high".to_string(),
            i: 0,
            doc_string: "high".to_string(),
            r#type: 1, // FLOAT
            f: high.unwrap_or(1.0),
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
        let att_shape = AttributeProto {
            name: "shape".to_string(),
            ref_attr_name: "shape".to_string(),
            i: 0,
            doc_string: "shape".to_string(),
            r#type: 7, // INTS
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: shape,
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let att_dtype = AttributeProto {
            name: "dtype".to_string(),
            ref_attr_name: "dtype".to_string(),
            i: 11, // DOUBLE
            doc_string: "dtype".to_string(),
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
            let mut mut_attrs = vec![att_shape, att_dtype];
            if low.is_some() {
                mut_attrs.push(att_low);
            }
            if high.is_some() {
                mut_attrs.push(att_high);
            }
            mut_attrs
        };
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "RandomUniform".to_string(),
                domain: "".to_string(),
                attribute: attrs,
                input: vec![],
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
        let eval = simple_eval(&manual_graph, HashMap::new())?;
        assert_eq!(eval.len(), 1);
        let z = eval.get("OUTPUT_Z").expect("Output 'z' not found");
        let min = z
            .flatten_all()?
            .to_vec1()?
            .into_iter()
            .reduce(f64::min)
            .unwrap();
        let max = z
            .flatten_all()?
            .to_vec1()?
            .into_iter()
            .reduce(f64::max)
            .unwrap();
        assert!(min >= low.unwrap_or(0.0).into());
        assert!(max <= high.unwrap_or(1.0).into());
        assert_ne!(min, max);
        Ok(())
    }

    Ok(())
}
