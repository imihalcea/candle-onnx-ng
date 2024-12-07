use candle_core::{DType, Device, NdArray, Tensor};
use candle_onnx_ng::onnx::{GraphProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

mod utils;

// "Range"
#[test]
fn test_range() -> candle_core::Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-113
    test(1., 5., 2., &[1., 3.])?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-113
    test(10i64, 6i64, -3i64, &[10i64, 7i64])?;

    fn test(
        start: impl NdArray,
        limit: impl NdArray,
        delta: impl NdArray,
        expected: impl NdArray,
    ) -> candle_core::Result<()> {
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Range".to_string(),
                domain: "".to_string(),
                attribute: vec![],
                input: vec![
                    "INPUT_X".to_string(),
                    "INPUT_Y".to_string(),
                    "INPUT_A".to_string(),
                ],
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
        inputs.insert("INPUT_X".to_string(), Tensor::new(start, &Device::Cpu)?);
        inputs.insert("INPUT_Y".to_string(), Tensor::new(limit, &Device::Cpu)?);
        inputs.insert("INPUT_A".to_string(), Tensor::new(delta, &Device::Cpu)?);

        let eval = simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval
            .get("OUTPUT_Z")
            .expect("Output 'z' not found")
            .to_dtype(DType::F64)?;

        let expected = Tensor::new(expected, &Device::Cpu)?.to_dtype(DType::F64)?;
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
