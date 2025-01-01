use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};

pub(crate) struct MaxPool;
impl OnnxOp for MaxPool {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool
        let dilations = node.get_attr_opt::<[i64]>("dilations")?;
        let kernel_shape = node.get_attr::<[i64]>("kernel_shape")?;
        let pads = node.get_attr_opt::<[i64]>("pads")?;
        let strides = node.get_attr_opt::<[i64]>("strides")?;
        let auto_pad = node.get_attr_opt::<str>("auto_pad")?;

        match auto_pad {
            None | Some("NOTSET") => (),
            Some(s) => {
                let error = format!("unsupported auto_pad {s}");
                return Err(OnnxOpError::UnsupportedAttribute(error));
            }
        };
        if let Some(d) = dilations {
            if d.iter().any(|&v| v != 1) {
                let error = format!("MaxPool with dilation != 1, {dilations:?}");
                return Err(OnnxOpError::UnsupportedAttribute(error));
            }
        }
        if let Some(d) = pads {
            if d.iter().any(|&v| v != 0) {
                let error = format!("MaxPool with pads != 0, {pads:?}");
                return Err(OnnxOpError::UnsupportedAttribute(error));
            }
        }
        let xs = node.get_input(0)?;

        let (k1, k2) = match kernel_shape {
            [k1, k2] => (*k1 as usize, *k2 as usize),
            _ => {
                let error = format!("only 2d MaxPool is supported, kernel shape {kernel_shape:?}");
                return Err(OnnxOpError::UnsupportedAttribute(error));
            }
        };

        let ys = match strides {
            None => xs.max_pool2d((k1, k2))?,
            Some([s1, s2]) => xs.max_pool2d_with_stride((k1, k2), (*s1 as usize, *s2 as usize))?,
            Some(strides) => {
                let error = format!("only 2d MaxPool is supported, strides {strides:?}");
                return Err(OnnxOpError::UnsupportedAttribute(error));
            }
        };

        let output_name = node.get_output(0)?;

        Ok(OpOutput::Single(output_name.clone(), ys))
    }
}
