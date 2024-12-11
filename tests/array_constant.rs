use candle_core::{DType, Device, NdArray, Tensor};
use candle_onnx_ng::onnx::attribute_proto::AttributeType;
use candle_onnx_ng::onnx::tensor_proto::DataType;
use candle_onnx_ng::onnx::{AttributeProto, GraphProto, NodeProto, TensorProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

mod utils;
#[test]
fn test_constant() -> candle_core::Result<()> {
    //https://github.com/onnx/onnx/blob/main/docs/Operators.md#Constant
    // "value" defaults to 0 f32
    test(&[1f32, 2., 3., 4.], &[1f32, 2., 3., 4.])?;
    test(&[1i64, 2, 3, 4], &[1i64, 2, 3, 4])?;

    fn test(value: impl NdArray, expected: impl NdArray) -> candle_core::Result<()> {
        let mut attribute = vec![];

        let tensor = Tensor::new(value, &Device::Cpu)?;

        let (data_type, raw_value) = match tensor.dtype() {
            DType::U8 => (
                DataType::Uint8,
                tensor
                    .to_vec1::<u8>()?
                    .iter()
                    .map(|x| x.to_le_bytes())
                    .flatten()
                    .collect(),
            ),
            DType::U32 => (
                DataType::Uint32,
                tensor
                    .to_vec1::<u32>()?
                    .iter()
                    .map(|x| x.to_le_bytes())
                    .flatten()
                    .collect(),
            ),
            DType::I64 => (
                DataType::Int64,
                tensor
                    .to_vec1::<i64>()?
                    .iter()
                    .map(|x| x.to_le_bytes())
                    .flatten()
                    .collect(),
            ),
            DType::F32 => (
                DataType::Float,
                tensor
                    .to_vec1::<f32>()?
                    .iter()
                    .map(|x| x.to_le_bytes())
                    .flatten()
                    .collect(),
            ),
            DType::F64 => (
                DataType::Double,
                tensor
                    .to_vec1::<f64>()?
                    .iter()
                    .map(|x| x.to_le_bytes())
                    .flatten()
                    .collect(),
            ),
            _ => panic!("unsupported DType in test"),
        };
        let tensor = TensorProto {
            data_type: data_type.into(),
            dims: tensor.dims().iter().map(|v| *v as i64).collect(),
            raw_data: raw_value,
            segment: None,
            float_data: vec![],
            int32_data: vec![],
            string_data: vec![],
            int64_data: vec![],
            name: "".to_string(),
            doc_string: "".to_string(),
            external_data: vec![],
            data_location: 0,
            double_data: vec![],
            uint64_data: vec![],
        };

        attribute.push(AttributeProto {
            name: "value".to_string(),
            ref_attr_name: "value".to_string(),
            i: 0,
            doc_string: "value".to_string(),
            r#type: AttributeType::Tensor.into(),
            f: 0.0,
            s: vec![],
            t: Some(tensor),
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
        });

        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Constant".to_string(),
                domain: "".to_string(),
                attribute,
                input: vec![],
                output: vec!["OUTPUT_Z".to_string()],
                name: "const_tensor".to_string(),
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

        let inputs: HashMap<String, Tensor> = HashMap::new();

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
