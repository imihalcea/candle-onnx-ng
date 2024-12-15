use candle_core::test_utils::to_vec2_round;
use candle_core::{bail, DType, Device, NdArray, Result, Tensor};
use candle_onnx_ng::onnx::attribute_proto::AttributeType;
use candle_onnx_ng::onnx::tensor_proto::DataType;
use candle_onnx_ng::onnx::tensor_shape_proto::{dimension, Dimension};
use candle_onnx_ng::onnx::{type_proto, TensorProto, TensorShapeProto, TypeProto};
use candle_onnx_ng::onnx::{AttributeProto, GraphProto, ModelProto, NodeProto, ValueInfoProto};
use candle_onnx_ng::simple_eval;
use std::collections::HashMap;

pub mod utils;

const INPUT_X: &str = "x";
const INPUT_Y: &str = "y";
const INPUT_A: &str = "a";
const OUTPUT_Z: &str = "z";

#[test]
fn test_evaluation_fails_without_defined_graph() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(None);
    let inputs: HashMap<String, Tensor> = HashMap::new();
    match simple_eval(&manual_graph, inputs) {
        Err(err) => assert_eq!(err.to_string(), "no graph defined in proto"),
        Ok(_) => panic!("Expected an error due to undefined graph"),
    }
    Ok(())
}

// Below are new_ops that are implemented but not tested yet

// "MaxPool"
// #[test]

// "AveragePool"
// #[test]

// "BatchNormalization"
// #[test]

// "Squeeze"
// #[test]

// "ConstantOfShape"

// "Unsqueeze"

// "Clip"
// #[test]

// "Shape"

// "Conv"
// #[test]

// "Concat"
// #[test]

// "Erf"
// #[test]

// "Constant"
// #[test]

// "Cast"
// #[test]

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
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
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
        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
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

// "RandomNormal"
#[test]
fn test_random_normal() -> Result<()> {
    test(vec![3, 2, 1, 4], None, None)?;
    test(vec![2, 2, 2, 2], Some(-10.0), None)?;
    test(vec![2, 2, 2, 2], None, Some(10.0))?;
    test(vec![1, 2, 3, 4], Some(-10.0), Some(10.0))?;

    fn test(shape: Vec<i64>, mean: Option<f32>, scale: Option<f32>) -> Result<()> {
        let att_mean = AttributeProto {
            name: "mean".to_string(),
            ref_attr_name: "mean".to_string(),
            i: 0,
            doc_string: "mean".to_string(),
            r#type: 1, // FLOAT
            f: mean.unwrap_or(0.0),
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
        let att_scale = AttributeProto {
            name: "scale".to_string(),
            ref_attr_name: "scale".to_string(),
            i: 0,
            doc_string: "scale".to_string(),
            r#type: 1, // FLOAT
            f: scale.unwrap_or(1.0),
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
            if mean.is_some() {
                mut_attrs.push(att_mean);
            }
            if scale.is_some() {
                mut_attrs.push(att_scale);
            }
            mut_attrs
        };
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "RandomNormal".to_string(),
                domain: "".to_string(),
                attribute: attrs,
                input: vec![],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
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

        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
        let data = z.flatten_all()?.to_vec1::<f64>()?;

        // test if values are unique
        for (i, a) in data.iter().enumerate() {
            for (j, b) in data.iter().enumerate() {
                if i == j {
                    continue;
                };
                assert_ne!(a, b);
            }
        }

        Ok(())
    }

    Ok(())
}

// "LeakyRelu"
#[test]
fn test_leakyrelu() -> Result<()> {
    // tests from https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-80
    // leakyrelu
    test(&[-1.0, 0.0, 1.0], Some(0.1), &[-0.1, 0.0, 1.0])?;
    fn test(data: impl NdArray, alpha: Option<f32>, expected: impl NdArray) -> Result<()> {
        let att_alpha = AttributeProto {
            name: "alpha".to_string(),
            ref_attr_name: "alpha".to_string(),
            i: 0,
            doc_string: "alpha".to_string(),
            r#type: 1, // FLOAT
            f: alpha.unwrap_or(0.01),
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
            let mut mut_attrs = vec![];
            if alpha.is_some() {
                mut_attrs.push(att_alpha);
            }
            mut_attrs
        };
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "LeakyRelu".to_string(),
                domain: "".to_string(),
                attribute: attrs,
                input: vec![INPUT_X.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));
        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(data, &Device::Cpu)?);
        let eval = simple_eval(&manual_graph, inputs)?;
        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;
        for both in z
            .to_vec1::<f64>()?
            .iter()
            .zip(expected.to_vec1::<f64>()?.iter())
        {
            let (act, exp) = both;
            assert!(f64::abs(act - exp) < f32::EPSILON.into());
        }

        Ok(())
    }

    Ok(())
}

// "If"
#[test]
fn test_if() -> Result<()> {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let output_type_proto = Some(TypeProto {
        value: Some(type_proto::Value::TensorType(type_proto::Tensor {
            elem_type: DataType::Float.into(),
            shape: Some(TensorShapeProto {
                dim: vec![Dimension {
                    denotation: "".to_string(),
                    value: Some(dimension::Value::DimValue(5)),
                }],
            }),
        })),
        denotation: "".to_string(),
    });
    let then_branch = GraphProto {
        output: vec![ValueInfoProto {
            name: "then_out".to_string(),
            r#type: output_type_proto.clone(),
            doc_string: "".to_string(),
        }],
        node: vec![NodeProto {
            op_type: "Constant".to_string(),
            input: vec![],
            output: vec!["then_out".to_string()],
            attribute: vec![AttributeProto {
                name: "value".to_string(),
                r#type: AttributeType::Tensor.into(),
                t: Some(TensorProto {
                    dims: vec![x.len() as i64],
                    float_data: x.clone(),
                    data_type: DataType::Float.into(),
                    ..TensorProto::default()
                }),
                ..AttributeProto::default()
            }],
            ..NodeProto::default()
        }],
        ..GraphProto::default()
    };
    let else_branch = GraphProto {
        output: vec![ValueInfoProto {
            name: "else_out".to_string(),
            r#type: output_type_proto.clone(),
            doc_string: "".to_string(),
        }],
        node: vec![NodeProto {
            op_type: "Constant".to_string(),
            input: vec![],
            output: vec!["else_out".to_string()],
            attribute: vec![AttributeProto {
                name: "value".to_string(),
                r#type: AttributeType::Tensor.into(),
                t: Some(TensorProto {
                    dims: vec![y.len() as i64],
                    float_data: y.clone(),
                    data_type: DataType::Float.into(),
                    ..TensorProto::default()
                }),
                ..AttributeProto::default()
            }],
            ..NodeProto::default()
        }],
        ..GraphProto::default()
    };
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "If".to_string(),
            attribute: vec![
                AttributeProto {
                    name: "then_branch".to_string(),
                    r#type: AttributeType::Graph.into(),
                    g: Some(then_branch),
                    ..AttributeProto::default()
                },
                AttributeProto {
                    name: "else_branch".to_string(),
                    r#type: AttributeType::Graph.into(),
                    g: Some(else_branch),
                    ..AttributeProto::default()
                },
            ],
            input: vec!["cond".to_string()],
            output: vec!["res".to_string()],
            ..NodeProto::default()
        }],
        input: vec![],
        output: vec![ValueInfoProto {
            name: "res".to_string(),
            doc_string: "".to_string(),
            r#type: output_type_proto.clone(),
        }],
        ..GraphProto::default()
    }));

    for cond in [1u8, 0] {
        let inputs =
            HashMap::from_iter([("cond".to_string(), Tensor::full(cond, (1,), &Device::Cpu)?)]);
        let outputs = simple_eval(&manual_graph, inputs)?;
        let expected = if cond != 0 { &x } else { &y };
        let Some(res) = outputs.get("res") else {
            bail!("outputs didn't contain expected key `res`: {outputs:?}");
        };
        assert_eq!(&res.to_vec1::<f32>()?, expected);
    }
    Ok(())
}

#[test]
fn test_lstm() -> Result<()> {
    // values generated from pytorch, so at least it's close enough to what pytorch does
    /*
    #!/usr/bin/env python3

    # torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)

    import torch

    rand_gen = torch.Generator()
    rand_gen.manual_seed(1)
    input_size = 3
    hidden_size = 5
    batch_size = 1
    sequence_length = 4
    number_directions = 1
    rnn = torch.nn.LSTM(input_size,hidden_size)
    weight_ih_l0 = torch.randn(rnn.weight_ih_l0.shape, generator=rand_gen)
    weight_hh_l0 = torch.randn(rnn.weight_hh_l0.shape, generator=rand_gen)
    bias_ih_l0 = torch.randn(rnn.bias_ih_l0.shape, generator=rand_gen)
    bias_hh_l0 = torch.randn(rnn.bias_hh_l0.shape, generator=rand_gen)
    rnn.weight_ih_l0 = torch.nn.Parameter(weight_ih_l0)
    rnn.weight_hh_l0 = torch.nn.Parameter(weight_hh_l0)
    rnn.bias_ih_l0 = torch.nn.Parameter(bias_ih_l0)
    rnn.bias_hh_l0 = torch.nn.Parameter(bias_hh_l0)
    input = torch.randn(sequence_length, batch_size, input_size, generator=rand_gen)
    h0 = torch.randn(number_directions, batch_size, hidden_size, generator=rand_gen)
    c0 = torch.randn(number_directions, batch_size, hidden_size, generator=rand_gen)
    output, (hn, cn) = rnn(input, (h0, c0))

    def fmt_tensor(t):
        return "Tensor::from_vec::<_, f32>(vec!"+  str(t.flatten().tolist()) + ", (" + "".join([str(n)+"," for n in t.shape])+"), &Device::Cpu)?"

    print("let input_size = ", input_size, ";")
    print("let hidden_size = ", hidden_size, ";")
    print("let batch_size = ", batch_size, ";")
    print("let sequence_length = ", sequence_length, ";")
    print("let number_directions = ", number_directions, ";")
    print("let weight_ih_l0 = ", fmt_tensor(rnn.weight_ih_l0), ";")
    print("let weight_hh_l0 = ", fmt_tensor(rnn.weight_hh_l0), ";")
    print("let bias_ih_l0 = ", fmt_tensor(rnn.bias_ih_l0), ";")
    print("let bias_hh_l0 = ", fmt_tensor(rnn.bias_hh_l0), ";")
    print("let input = ", fmt_tensor(input), ";")
    print("let h0 = ", fmt_tensor(h0), ";")
    print("let c0 = ", fmt_tensor(c0), ";")
    print("let output = ", fmt_tensor(output), ";")
    print("let hn = ", fmt_tensor(hn), ";")
    print("let cn = ", fmt_tensor(cn), ";")
    */
    let input_size = 3;
    let hidden_size = 5;
    let batch_size = 1;
    let sequence_length = 4;
    let number_directions = 1;
    let weight_ih_l0 = Tensor::from_vec::<_, f32>(
        vec![
            -1.525_595_9,
            -0.750_231_8,
            -0.653_980_9,
            -1.609_484_8,
            -0.100_167_18,
            -0.609_188_9,
            -0.979_772_27,
            -1.609_096_3,
            -0.712_144_6,
            0.303_722,
            -0.777_314_3,
            -0.251_455_25,
            -0.222_270_49,
            1.687_113_4,
            0.228_425_17,
            0.467_635_5,
            -0.696_972_4,
            -1.160_761_5,
            0.699_542_4,
            0.199_081_63,
            0.865_692_4,
            0.244_403_9,
            -0.662_911_36,
            0.807_308_26,
            1.101_680_6,
            -0.175_936_04,
            -2.245_557_8,
            -1.446_458,
            0.061_155_282,
            -0.617_744_45,
            -0.798_069_83,
            -0.131_623_21,
            1.879_345_8,
            -0.072_131_78,
            0.157_770_6,
            -0.773_454_9,
            0.199_056_5,
            0.045_702_778,
            0.152_956_92,
            -0.475_678_8,
            -0.111_019_83,
            0.292_735_25,
            -0.157_845_15,
            -0.028_787_14,
            0.453_254_58,
            1.142_161_1,
            0.248_610_7,
            -1.775_400_8,
            -0.025_502_462,
            -1.023_330_6,
            -0.596_185_15,
            -1.005_530_7,
            0.428_542_3,
            1.476_077_8,
            -1.786_867_9,
            1.610_317_6,
            -0.703_956_66,
            -0.185_265_8,
            -0.996_235_1,
            -0.831_255_26,
        ],
        (20, 3),
        &Device::Cpu,
    )?;
    let weight_hh_l0 = Tensor::from_vec::<_, f32>(
        vec![
            0.409_972_43,
            0.408_450_66,
            0.257_865_4,
            1.095_021_4,
            -0.506_486_6,
            0.099_775_404,
            -0.653_973_4,
            0.731_693_7,
            -1.456_733,
            1.608_935_4,
            0.093_769_975,
            -1.259_749,
            0.254_633_5,
            -0.501_957_3,
            -1.041_2,
            0.732_267_2,
            1.307_535_5,
            -1.162_798_8,
            0.119_636_11,
            -0.163_135_33,
            0.661_445_3,
            1.189_920_5,
            0.816_533_9,
            -0.913_523_6,
            -0.353_806_53,
            0.763_927_04,
            -0.588_950_7,
            -0.763_597_37,
            1.335_205_7,
            0.604_273_6,
            -0.103_442_08,
            -0.151_216_92,
            1.246_568_3,
            0.505_721_4,
            0.950_511_2,
            1.296_648_3,
            0.873_796_3,
            -0.560_259_4,
            1.285_784_5,
            0.816_823_84,
            -1.464_799_4,
            -1.262_928_4,
            1.122_018_8,
            1.566_334_1,
            2.558_138_4,
            -0.233_363_88,
            -0.013_472_13,
            1.860_634_8,
            1.549_620_5,
            0.347_629_25,
            0.093_008_03,
            0.614_740_3,
            0.712_364_55,
            -1.776_507_3,
            0.353_864_58,
            1.199_613_2,
            -0.712_258_93,
            -0.620_034_4,
            -0.228_134_95,
            -0.789_274_63,
            -1.611_111_8,
            -1.871_612_9,
            0.543_083_6,
            0.660_678_6,
            0.270_527_72,
            0.559_691_97,
            -0.318_396_3,
            1.511_720_7,
            -1.363_267_2,
            -0.983_219_6,
            1.511_266_7,
            0.641_870_74,
            -0.747_445_9,
            -0.923_438_55,
            0.573_398_4,
            -0.109_299_51,
            0.518_112_1,
            0.106_535_35,
            0.269_240_77,
            1.324_768,
            0.037_456_9,
            -0.637_839_3,
            -0.814_755_44,
            -0.689_506_53,
            0.843_654_3,
            1.165_701_3,
            0.526_932_2,
            1.619_253_3,
            -0.963_976_26,
            0.141_520_38,
            -0.163_660_96,
            -0.358_222_57,
            1.722_279_3,
            -0.303_575_6,
            0.238_874_2,
            1.344_001_2,
            0.103_225_69,
            1.100_354_2,
            -0.341_680_2,
            0.947_338_9,
        ],
        (20, 5),
        &Device::Cpu,
    )?;
    let bias_ih_l0 = Tensor::from_vec::<_, f32>(
        vec![
            -0.568_515_96,
            0.837_596_2,
            1.783_660_7,
            -0.195_424_66,
            0.235_193_13,
            1.914_243_3,
            1.836_411_1,
            1.324_532_4,
            -0.070_514_58,
            0.346_979_4,
            -0.653_679_6,
            1.558_620_2,
            0.218_566_15,
            -0.574_307_26,
            1.457_125_1,
            1.770_955_7,
            -2.017_3,
            0.423_503_2,
            0.573_022,
            -1.796_243,
        ],
        (20,),
        &Device::Cpu,
    )?;
    let bias_hh_l0 = Tensor::from_vec::<_, f32>(
        vec![
            1.247_040_4,
            1.273_851_2,
            0.390_949_25,
            0.387_210_5,
            0.144_403_95,
            0.777_168_45,
            -2.338_112_6,
            -0.829_120_4,
            1.166_139_1,
            1.478_657_5,
            0.267_608_73,
            0.756_119_85,
            -0.587_336_1,
            -2.061_920_6,
            0.430_473_48,
            0.337_656_62,
            -0.343_785_35,
            -0.617_226_06,
            1.252_969_3,
            -0.051_417_42,
        ],
        (20,),
        &Device::Cpu,
    )?;
    let input = Tensor::from_vec::<_, f32>(
        vec![
            0.647_212_8,
            -0.041_167_17,
            -0.177_493_08,
            -0.500_039_3,
            0.867_274_94,
            -0.273_192_23,
            -0.460_768_13,
            -0.099_093_71,
            0.472_844_8,
            1.004_948_5,
            -0.287_142_04,
            -1.161_862_1,
        ],
        (4, 1, 3),
        &Device::Cpu,
    )?;
    let h0 = Tensor::from_vec::<_, f32>(
        vec![
            0.027_581_785,
            0.565_238_24,
            -0.011_487_379,
            0.670_640_05,
            -0.492_925_05,
        ],
        (1, 1, 5),
        &Device::Cpu,
    )?;
    let c0 = Tensor::from_vec::<_, f32>(
        vec![
            1.505_028_5,
            -2.326_355,
            1.616_89,
            -0.902_623_8,
            0.173_668_24,
        ],
        (1, 1, 5),
        &Device::Cpu,
    )?;
    let output = Tensor::from_vec::<_, f32>(
        vec![
            0.595_601_7,
            -0.017_232_792,
            0.110_355_72,
            -0.493_231_74,
            0.047_632_16,
            0.635_845_2,
            0.040_328_12,
            -0.378_861_16,
            -0.746_434,
            0.200_809_09,
            0.584_026_5,
            0.145_328_82,
            -0.734_529_85,
            -0.521_430_43,
            0.219_038_17,
            0.742_045_16,
            0.319_438_8,
            -0.047_266_465,
            -0.282_384_96,
            0.271_313_4,
        ],
        (4, 1, 5),
        &Device::Cpu,
    )?;
    let hn = Tensor::from_vec::<_, f32>(
        vec![
            0.742_045_16,
            0.319_438_8,
            -0.047_266_465,
            -0.282_384_96,
            0.271_313_4,
        ],
        (1, 1, 5),
        &Device::Cpu,
    )?;
    let cn = Tensor::from_vec::<_, f32>(
        vec![
            0.963_055_85,
            1.003_307,
            -1.754_899,
            -1.596_712_2,
            0.825_292_47,
        ],
        (1, 1, 5),
        &Device::Cpu,
    )?;
    // end of generated values

    let model = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "LSTM".to_string(),
            name: "LSTM_test".to_string(),
            attribute: vec![AttributeProto {
                name: "hidden_size".to_string(),
                r#type: AttributeType::Int.into(),
                i: hidden_size as i64,
                ..AttributeProto::default()
            }],
            input: vec![
                "input".to_string(),
                "w".to_string(),
                "r".to_string(),
                "b".to_string(), // b
                "".to_string(),  // seq_lens
                "h".to_string(),
                "c".to_string(),
            ],
            output: vec!["output".to_string(), "hn".to_string(), "cn".to_string()],
            ..NodeProto::default()
        }],
        input: ["input", "w", "r", "b", "h", "c"]
            .into_iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                ..ValueInfoProto::default()
            })
            .collect(),
        output: ["output", "hn", "cn"]
            .into_iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                ..ValueInfoProto::default()
            })
            .collect(),
        ..GraphProto::default()
    }));
    // pytorch stores weight and bias as [ifco] but we want it as [iofc]
    // so we need to re-arrange the tensors a bit
    let idx_iofc = {
        let stride = hidden_size as i64;
        let dev = weight_ih_l0.device();
        let idx_i = Tensor::arange(0, stride, dev)?;
        let idx_f = Tensor::arange(stride, 2 * stride, dev)?;
        let idx_g = Tensor::arange(2 * stride, 3 * stride, dev)?;
        let idx_o = Tensor::arange(3 * stride, 4 * stride, dev)?;

        Tensor::cat(&[&idx_i, &idx_o, &idx_f, &idx_g], 0)?
    };
    let w = weight_ih_l0.index_select(&idx_iofc, 0)?;
    let w = w.reshape((number_directions, 4 * hidden_size, input_size))?;
    let r = weight_hh_l0.index_select(&idx_iofc, 0)?;
    let r = r.reshape((number_directions, 4 * hidden_size, hidden_size))?;
    let wb = bias_ih_l0.index_select(&idx_iofc, 0)?;
    let rb = bias_hh_l0.index_select(&idx_iofc, 0)?;
    let b = Tensor::cat(&[wb, rb], 0)?.reshape((number_directions, 8 * hidden_size))?;
    let output = output.reshape((sequence_length, number_directions, batch_size, hidden_size))?;
    let result = simple_eval(
        &model,
        HashMap::from_iter([
            ("input".to_string(), input),
            ("w".to_string(), w),
            ("r".to_string(), r),
            ("b".to_string(), b),
            ("h".to_string(), h0),
            ("c".to_string(), c0),
        ]),
    )?;
    let actual_output = result.get("output").unwrap();
    assert_eq!(output.dims(), actual_output.dims());
    let actual_hn = result.get("hn").unwrap();
    assert_eq!(hn.dims(), actual_hn.dims());
    let actual_cn = result.get("cn").unwrap();
    assert_eq!(cn.dims(), actual_cn.dims());
    let diff_close_enough = |a: &Tensor, b| -> Result<_> {
        let diffs = a.sub(b)?.flatten_all()?.to_vec1::<f32>()?;
        Ok(diffs.iter().all(|f| f.abs() < 0.0001))
    };
    assert!(
        diff_close_enough(&output, actual_output)?,
        "output did not match expected\n{actual_output}\n{output}",
    );
    assert!(
        diff_close_enough(&hn, actual_hn)?,
        "hn did not match expected\n{actual_hn}\n{hn}",
    );
    assert!(
        diff_close_enough(&cn, actual_cn)?,
        "cn did not match expected\n{actual_cn}\n{cn}",
    );

    Ok(())
}

fn make_split_graph_helper(inputs: &[&str], outputs: &[&str], axis: i64) -> ModelProto {
    let attribs = vec![AttributeProto {
        name: "axis".to_string(),
        r#type: AttributeType::Int.into(),
        i: axis,
        ..AttributeProto::default()
    }];

    make_graph_helper("Split", inputs, outputs, attribs)
}

#[test]
fn test_split_equal_parts_1d_opset13() -> Result<()> {
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
