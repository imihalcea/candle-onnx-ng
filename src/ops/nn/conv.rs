use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};

pub(crate) struct Conv;

impl OnnxOp for Conv {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
        let dilations = node.get_attr_opt::<[i64]>("dilations")?;
        let groups = node.get_attr_opt::<i64>("group")?.copied().unwrap_or(1);
        let _kernel_shape = node.get_attr_opt::<[i64]>("kernel_shape")?;
        let pads = node.get_attr_opt::<[i64]>("pads")?;
        let strides = node.get_attr_opt::<[i64]>("strides")?;
        let auto_pad = node.get_attr_opt::<str>("auto_pad")?;
        match auto_pad {
            None | Some("NOTSET") => (),
            Some(s) => {
                return Err(OnnxOpError::UnsupportedAttribute(format!(
                    "unsupported auto_pad {s}"
                )))
            }
        };
        let xs = node.get_input(0)?;
        let ws = node.get_input(1)?;
        let ys = match ws.rank() {
            3 => {
                let (pads, xs) = match pads {
                    None => (0, xs.clone()),
                    Some([p]) => (*p as usize, xs.clone()),
                    Some([p1, p2]) => {
                        if p1 != p2 {
                            (0usize, xs.pad_with_zeros(2, *p1 as usize, *p2 as usize)?)
                        } else {
                            (*p1 as usize, xs.clone())
                        }
                    }
                    Some(pads) => {
                        return Err(OnnxOpError::ComputationFailed(format!(
                            "more pads than expected in conv1d {pads:?} {}",
                            node.name
                        )))
                    }
                };
                let strides = match strides {
                    None => 1,
                    Some([p]) => *p as usize,
                    Some(s) => {
                        return Err(OnnxOpError::ComputationFailed(format!(
                            "more strides than expected in conv1d {s:?} {}",
                            node.name
                        )))
                    }
                };
                let dilations = match dilations {
                    None => 1,
                    Some([p]) => *p as usize,
                    Some(s) => {
                        return Err(OnnxOpError::ComputationFailed(format!(
                            "more dilations than expected in conv1d {s:?} {}",
                            node.name
                        )))
                    }
                };
                xs.conv1d(ws, pads, strides, dilations, groups as usize)?
            }
            4 => {
                let (pads, xs) = match pads {
                    None => (0, xs.clone()),
                    Some([p]) => (*p as usize, xs.clone()),
                    Some(&[p1, p2, p3, p4]) => {
                        let p1 = p1 as usize;
                        let p2 = p2 as usize;
                        let p3 = p3 as usize;
                        let p4 = p4 as usize;
                        if p1 != p2 || p1 != p3 || p1 != p4 {
                            (0, xs.pad_with_zeros(2, p1, p3)?.pad_with_zeros(3, p2, p4)?)
                        } else {
                            (p1, xs.clone())
                        }
                    }
                    Some(pads) => {
                        return Err(OnnxOpError::ComputationFailed(format!(
                            "more pads than expected in conv2d {pads:?} {}",
                            node.name
                        )))
                    }
                };
                let strides = match strides {
                    None => 1,
                    Some([p]) => *p as usize,
                    Some([p1, p2]) => {
                        if p1 != p2 {
                            return Err(OnnxOpError::ComputationFailed(format!(
                                "strides have to be the same on both axis {pads:?} {}",
                                node.name
                            )));
                        }
                        *p1 as usize
                    }
                    Some(s) => {
                        return Err(OnnxOpError::ComputationFailed(format!(
                            "more strides than expected in conv2d {s:?} {}",
                            node.name
                        )))
                    }
                };
                let dilations = match dilations {
                    None => 1,
                    Some([p]) => *p as usize,
                    Some([p1, p2]) => {
                        if p1 != p2 {
                            return Err(OnnxOpError::ComputationFailed(format!(
                                "dilations have to be the same on both axis {pads:?} {}",
                                node.name
                            )));
                        }
                        *p1 as usize
                    }
                    Some(s) => {
                        return Err(OnnxOpError::ComputationFailed(format!(
                            "more dilations than expected in conv2d {s:?} {}",
                            node.name
                        )))
                    }
                };
                xs.conv2d(ws, pads, strides, dilations, groups as usize)?
            }
            rank => {
                return Err(OnnxOpError::ComputationFailed(format!(
                    "unsupported rank for weight matrix {rank} in conv {}",
                    node.name
                )))
            }
        };
        let ys = if node.input_len() > 2 {
            let bs = node.get_input(2)?;
            let mut bs_shape = vec![1; ys.rank()];
            bs_shape[1] = bs.elem_count();
            ys.broadcast_add(&bs.reshape(bs_shape)?)?
        } else {
            ys
        };

        let output_name = node.get_output(0)?;
        Ok(OpOutput::Single(output_name.clone(), ys))
    }
}
