use crate::onnx::tensor_proto::DataType;
use crate::ops::OnnxOpError::InvalidAttribute;
use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};
use crate::parser;
use candle_core::{DType, Device, Tensor};

fn validate_dtype_value(
    random_type: &str,
    node: &ComputeNode,
    dt: i64,
) -> Result<DType, Result<OpOutput, OnnxOpError>> {
    Ok(match DataType::try_from(dt as i32) {
        Ok(dt) => match parser::dtype(dt) {
            Some(DType::U8 | DType::U32 | DType::I64) => {
                let msg = format!(
                    "unsupported 'dtype' value {dt:?}, only floats are allowed, for {random_type} {}",
                    node.name
                );
                return Err(Err(InvalidAttribute(msg)));
            }
            Some(dt) => dt,
            None => {
                let msg = format!(
                    "unsupported 'dtype' value {dt:?} for {random_type} {}",
                    node.name
                );
                return Err(Err(InvalidAttribute(msg)));
            }
        },
        Err(_) => {
            let msg = format!(
                "unsupported 'dtype' value {dt:?} for {random_type} {}",
                node.name
            );
            return Err(Err(InvalidAttribute(msg)));
        }
    })
}

pub(crate) struct RandomNormal;
impl OnnxOp for RandomNormal {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let dt: i64 = node.get_attr_opt("dtype")?.copied().unwrap_or(1);
        // 1 is float type by default
        let dtype = match validate_dtype_value("RandomNormal", node, dt) {
            Ok(value) => value,
            Err(value) => return value,
        };
        let seed: Option<f32> = node.get_attr_opt("seed")?.copied();
        if seed.is_some() {
            let msg = "seed for RandomNormal is currently not supported".to_string();
            return Err(OnnxOpError::UnsupportedAttribute(msg));
        };
        let shape: Vec<usize> = node
            .get_attr::<[i64]>("shape")?
            .iter()
            .map(|x| *x as usize)
            .collect();

        let mean: f32 = node.get_attr_opt("mean")?.copied().unwrap_or(0.0);
        let scale: f32 = node.get_attr_opt("scale")?.copied().unwrap_or(1.0);

        let output = Tensor::randn(mean, scale, shape, &Device::Cpu)?.to_dtype(dtype)?;
        let output_name = node.get_output(0)?.clone();
        Ok(OpOutput::Single(output_name, output))
    }
}

pub(crate) struct RandomUniform;
impl OnnxOp for RandomUniform {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let dt: i64 = node.get_attr_opt("dtype")?.copied().unwrap_or(1);
        // 1 is float type by default
        let dtype = match validate_dtype_value("RandomUniform", node, dt) {
            Ok(value) => value,
            Err(value) => return value,
        };
        let seed: Option<f32> = node.get_attr_opt("seed")?.copied();
        if seed.is_some() {
            let msg = "seed for RandomUniform is currently not supported".to_string();
            return Err(OnnxOpError::UnsupportedAttribute(msg));
        };
        let shape: Vec<usize> = node
            .get_attr::<[i64]>("shape")?
            .iter()
            .map(|x| *x as usize)
            .collect();

        let low: f32 = node.get_attr_opt("low")?.copied().unwrap_or(0.0);
        let high: f32 = node.get_attr_opt("high")?.copied().unwrap_or(1.0);

        let output = Tensor::rand(low, high, shape, &Device::Cpu)?.to_dtype(dtype)?;
        let output_name = node.get_output(0)?.clone();
        Ok(OpOutput::Single(output_name, output))
    }
}
