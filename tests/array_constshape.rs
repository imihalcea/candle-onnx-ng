use candle_core::{DType, Device, NdArray, Tensor};
use candle_onnx_ng::onnx::attribute_proto::AttributeType;
use candle_onnx_ng::onnx::tensor_proto::DataType;
use candle_onnx_ng::onnx::{AttributeProto, GraphProto, NodeProto, TensorProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

mod utils;
#[test]
fn test_constant_of_shape() -> candle_core::Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-31
    test(&[4i64, 3, 2], Some(1.), &[1., 1., 1.])?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-31
    test(&[0.], Some(0i64), &[0i64])?;

    // "value" defaults to 0 f32
    test(&[1i64, 2, 3, 4], None as Option<i64>, &[0., 0., 0., 0.])?;

    fn test(
        input: impl NdArray,
        value: Option<impl NdArray>,
        expected: impl NdArray,
    ) -> candle_core::Result<()> {
        let mut attribute = vec![];

        if let Some(value) = value {
            let tensor = Tensor::new(value, &Device::Cpu)?;

            let (value, data_type) = match tensor.dtype() {
                DType::U8 => (
                    tensor.to_vec0::<u8>()?.to_le_bytes().to_vec(),
                    DataType::Uint8,
                ),
                DType::U32 => (
                    tensor.to_vec0::<u32>()?.to_le_bytes().to_vec(),
                    DataType::Uint32,
                ),
                DType::I64 => (
                    tensor.to_vec0::<i64>()?.to_le_bytes().to_vec(),
                    DataType::Int64,
                ),
                DType::F32 => (
                    tensor.to_vec0::<f32>()?.to_le_bytes().to_vec(),
                    DataType::Float,
                ),
                DType::F64 => (
                    tensor.to_vec0::<f64>()?.to_le_bytes().to_vec(),
                    DataType::Double,
                ),
                _ => panic!("unsupported DType in test"),
            };
            let tensor = TensorProto {
                data_type: data_type.into(),
                dims: tensor.dims().iter().map(|v| *v as i64).collect(),
                raw_data: value,
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
            })
        }

        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "ConstantOfShape".to_string(),
                domain: "".to_string(),
                attribute,
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
        inputs.insert("INPUT_X".to_string(), Tensor::new(input, &Device::Cpu)?);

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
