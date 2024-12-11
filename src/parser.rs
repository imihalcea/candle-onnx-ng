use crate::onnx;
use crate::onnx::attribute_proto::AttributeType;
use crate::onnx::tensor_proto::DataType;
use crate::onnx::GraphProto;
use candle_core::{bail, DType, Device, Tensor};

pub type Value = Tensor;

pub fn dtype(dt: DataType) -> Option<DType> {
    match dt {
        DataType::Uint8 => Some(DType::U8),
        DataType::Uint32 => Some(DType::U32),
        DataType::Int64 => Some(DType::I64),
        DataType::Float16 => Some(DType::F16),
        DataType::Float => Some(DType::F32),
        DataType::Double => Some(DType::F64),
        DataType::Bool => Some(DType::U8),
        _ => None,
    }
}

pub(crate) trait Attr {
    const TYPE: AttributeType;
    fn get(attr: &onnx::AttributeProto) -> candle_core::Result<&Self>;
}

pub(crate) trait AttrOwned: Sized {
    const TYPE: AttributeType;
    fn get(attr: &onnx::AttributeProto) -> candle_core::Result<Self>;
}

impl Attr for i64 {
    const TYPE: AttributeType = AttributeType::Int;
    fn get(attr: &onnx::AttributeProto) -> candle_core::Result<&Self> {
        Ok(&attr.i)
    }
}

impl Attr for f32 {
    const TYPE: AttributeType = AttributeType::Float;
    fn get(attr: &onnx::AttributeProto) -> candle_core::Result<&Self> {
        Ok(&attr.f)
    }
}

impl Attr for [i64] {
    const TYPE: AttributeType = AttributeType::Ints;
    fn get(attr: &onnx::AttributeProto) -> candle_core::Result<&Self> {
        Ok(attr.ints.as_slice())
    }
}

impl Attr for str {
    const TYPE: AttributeType = AttributeType::String;
    fn get(attr: &onnx::AttributeProto) -> candle_core::Result<&Self> {
        std::str::from_utf8(&attr.s).map_err(candle_core::Error::wrap)
    }
}

impl Attr for GraphProto {
    const TYPE: AttributeType = AttributeType::Graph;
    fn get(attr: &onnx::AttributeProto) -> candle_core::Result<&Self> {
        attr.g
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("attribute does not contain graph".to_string()))
    }
}

impl AttrOwned for Vec<String> {
    const TYPE: AttributeType = AttributeType::Strings;
    fn get(attr: &onnx::AttributeProto) -> candle_core::Result<Self> {
        let mut ret = vec![];
        for bytes in attr.strings.iter() {
            let s = String::from_utf8(bytes.clone()).map_err(candle_core::Error::wrap)?;
            ret.push(s);
        }
        Ok(ret)
    }
}

impl AttrOwned for Tensor {
    const TYPE: AttributeType = AttributeType::Tensor;
    fn get(attr: &onnx::AttributeProto) -> candle_core::Result<Self> {
        let tensor_proto = match &attr.t {
            Some(value) => value,
            None => bail!(
                "attribute {} was of type TENSOR, but no tensor was found",
                attr.name
            ),
        };

        let data_type = match DataType::try_from(tensor_proto.data_type) {
            Ok(value) => value,
            Err(_) => bail!(
                "attribute {} of type TENSOR was an invalid data_type number {}",
                attr.name,
                tensor_proto.data_type
            ),
        };

        let dtype = match dtype(data_type) {
            Some(value) => value,
            None => bail!(
                "attribute {} of type TENSOR has an unsupported data_type {}",
                attr.name,
                data_type.as_str_name()
            ),
        };

        let mut dims = Vec::with_capacity(tensor_proto.dims.len());
        for dim in &tensor_proto.dims {
            if dim < &0 {
                bail!(
                    "attribute {} of type TENSOR has a negative dimension, which is unsupported",
                    attr.name
                )
            }
            dims.push(*dim as usize)
        }

        Tensor::from_raw_buffer(&tensor_proto.raw_data, dtype, &dims, &Device::Cpu)
    }
}

fn get_attr_<'a>(
    node: &'a onnx::NodeProto,
    name: &str,
) -> candle_core::Result<&'a onnx::AttributeProto> {
    match node.attribute.iter().find(|attr| attr.name == name) {
        None => {
            bail!(
                "cannot find the '{name}' attribute in '{}' for {}",
                node.op_type,
                node.name
            )
        }
        Some(dt) => Ok(dt),
    }
}

pub fn get_attr<'a, T: Attr + ?Sized>(
    node: &'a onnx::NodeProto,
    name: &str,
) -> candle_core::Result<&'a T> {
    let attr = get_attr_(node, name)?;
    if attr.r#type() != T::TYPE {
        bail!(
            "unsupported type {:?} for '{name}' attribute in '{}' for {}",
            attr.r#type,
            node.op_type,
            node.name
        )
    }
    T::get(attr)
}

pub fn get_attr_opt<'a, T: Attr + ?Sized>(
    node: &'a onnx::NodeProto,
    name: &str,
) -> candle_core::Result<Option<&'a T>> {
    match node.attribute.iter().find(|attr| attr.name == name) {
        None => Ok(None),
        Some(attr) => {
            if attr.r#type() != T::TYPE {
                bail!(
                    "unsupported type {:?} for '{name}' attribute in '{}' for {}",
                    attr.r#type,
                    node.op_type,
                    node.name
                )
            }
            let val = T::get(attr)?;
            Ok(Some(val))
        }
    }
}

pub fn get_attr_definition<'a>(
    node: &'a onnx::NodeProto,
    name: &str,
) -> Option<&'a onnx::AttributeProto> {
    node.attribute.iter().find(|attr| attr.name == name)
}

pub fn get_attr_opt_owned<T: AttrOwned>(
    node: &onnx::NodeProto,
    name: &str,
) -> candle_core::Result<Option<T>> {
    match node.attribute.iter().find(|attr| attr.name == name) {
        None => Ok(None),
        Some(attr) => {
            if attr.r#type() != T::TYPE {
                bail!(
                    "unsupported type {:?} for '{name}' attribute in '{}' for {}",
                    attr.r#type,
                    node.op_type,
                    node.name
                )
            }
            let val = T::get(attr)?;
            Ok(Some(val))
        }
    }
}

pub fn get_tensor(t: &onnx::TensorProto, name: &str) -> candle_core::Result<Tensor> {
    let dims: Vec<usize> = t.dims.iter().map(|&x| x as usize).collect();
    match DataType::try_from(t.data_type) {
        Ok(DataType::Int32) => {
            if t.int32_data.is_empty() {
                let len = t.raw_data.len() / 4;
                let data: &[i32] =
                    unsafe { std::slice::from_raw_parts(t.raw_data.as_ptr() as *const i32, len) };
                let data = data.iter().map(|v| *v as i64).collect::<Vec<_>>();
                Tensor::from_vec(data, len, &Device::Cpu)
            } else {
                let data = t.int32_data.iter().map(|v| *v as i64).collect::<Vec<_>>();
                Tensor::from_vec(data, t.int32_data.len(), &Device::Cpu)
            }
        }
        Ok(dt) => match dtype(dt) {
            Some(dt) => {
                if dt == DType::F32 && !t.float_data.is_empty() {
                    Tensor::from_slice(&t.float_data, dims.as_slice(), &Device::Cpu)
                } else if dt == DType::F64 && !t.double_data.is_empty() {
                    Tensor::from_slice(&t.double_data, dims.as_slice(), &Device::Cpu)
                } else if dt == DType::I64 && !t.int64_data.is_empty() {
                    Tensor::from_slice(&t.int64_data, dims.as_slice(), &Device::Cpu)
                } else {
                    Tensor::from_raw_buffer(
                        t.raw_data.as_slice(),
                        dt,
                        dims.as_slice(),
                        &Device::Cpu,
                    )
                }
            }
            None => {
                bail!("unsupported 'value' data-type {dt:?} for {name}")
            }
        },
        Err(_) => {
            bail!("unsupported 'value' data-type {} for {name}", t.data_type,)
        }
    }
}
