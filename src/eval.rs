use crate::onnx::attribute_proto::AttributeType;
use crate::onnx::tensor_proto::DataType;
use crate::onnx::{self, GraphProto};
use crate::ops::{registry, ComputeNode};
use candle_core as candle;
use candle_core::{bail, DType, Device, Result, Tensor};

use crate::parser;
use crate::parser::Value;
use std::collections::{HashMap, HashSet};

// This function provides a direct evaluation of the proto.
// Longer-term, we should first convert the proto to an intermediate representation of the compute
// graph so as to make multiple evaluations more efficient.
// An example upside of this would be to remove intermediary values when they are not needed
// anymore.
pub fn simple_eval(
    model: &onnx::ModelProto,
    mut inputs: HashMap<String, Value>,
) -> Result<HashMap<String, Value>> {
    let graph = match &model.graph {
        None => bail!("no graph defined in proto"),
        Some(graph) => graph,
    };
    simple_eval_(graph, &mut inputs)
}

fn simple_eval_(
    graph: &onnx::GraphProto,
    values: &mut HashMap<String, Value>,
) -> Result<HashMap<String, Value>> {
    for t in graph.initializer.iter() {
        let tensor = parser::get_tensor(t, t.name.as_str())?;
        values.insert(t.name.to_string(), tensor);
    }
    for input in graph.input.iter() {
        let input_type = match &input.r#type {
            Some(input_type) => input_type,
            None => continue,
        };
        let input_type = match &input_type.value {
            Some(input_type) => input_type,
            None => continue,
        };
        let tensor_type = match input_type {
            onnx::type_proto::Value::TensorType(tt) => tt,
            _ => continue,
        };

        let tensor = match values.get(&input.name) {
            None => bail!("missing input {}", input.name),
            Some(tensor) => tensor,
        };
        let dt = match DataType::try_from(tensor_type.elem_type) {
            Ok(dt) => match parser::dtype(dt) {
                Some(dt) => dt,
                None => {
                    bail!("unsupported 'value' data-type {dt:?} for {}", input.name)
                }
            },
            type_ => bail!("unsupported input type {type_:?}"),
        };
        match &tensor_type.shape {
            None => continue,
            Some(shape) => {
                if shape.dim.len() != tensor.rank() {
                    bail!(
                        "unexpected rank for {}, got {:?}, expected {:?}",
                        input.name,
                        shape.dim,
                        tensor.shape()
                    )
                }
                for (idx, (d, &dim)) in shape.dim.iter().zip(tensor.dims().iter()).enumerate() {
                    match &d.value {
                        Some(onnx::tensor_shape_proto::dimension::Value::DimValue(v)) => {
                            if *v as usize != dim {
                                bail!(
                                    "unexpected dim {idx} for {}, got {:?}, expected {:?}",
                                    input.name,
                                    shape.dim,
                                    tensor.shape()
                                )
                            }
                        }
                        // We do not check equality constraints for the DimParam dimensions for now.
                        Some(onnx::tensor_shape_proto::dimension::Value::DimParam(_)) | None => (),
                    }
                }
            }
        };
        if dt != tensor.dtype() {
            bail!(
                "unexpected dtype for {}, got {:?}, expected {dt:?}",
                input.name,
                tensor.dtype()
            )
        }
    }

    let registry = registry()?;
    // The nodes are topologically sorted so we can just process them in order.
    for node in graph.node.iter() {
        let get = |input_name: &str| match values.get(input_name) {
            Some(value) => Ok(value),
            None => bail!("cannot find {input_name} for op '{}'", node.name),
        };
        let get_opt = |i: usize| {
            node.input
                .get(i)
                .filter(|s: &&String| !s.is_empty())
                .map(|s| get(s))
        };

        // TODO: Validate node.input for each operator.
        match node.op_type.as_str() {
            "Floor" => {
                let input = get(&node.input[0])?;
                let output = input.floor()?;
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Constant
            "Constant" => {
                let value = match node.attribute.iter().find(|attr| attr.name == "value") {
                    None => {
                        // TODO: support sparse_value etc.
                        bail!("cannot find 'value' attr in 'Constant' for {}", node.name)
                    }
                    Some(value) => value,
                };
                let output = match value.r#type() {
                    AttributeType::Tensor => {
                        let t = value.t.as_ref().unwrap();
                        parser::get_tensor(t, &node.name)?
                    }
                    rtype => bail!("unsupported 'value' type {rtype:?} for {}", node.name),
                };

                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast
            "Cast" => {
                let input = get(&node.input[0])?;
                let dt: i64 = *parser::get_attr(node, "to")?;
                let dtype = match DataType::try_from(dt as i32) {
                    Ok(DataType::Int32) => DType::I64,
                    Ok(dt) => match parser::dtype(dt) {
                        Some(dt) => dt,
                        None => {
                            bail!("unsupported 'to' value {dt:?} for cast {}", node.name)
                        }
                    },
                    Err(_) => {
                        bail!("unsupported 'to' value {dt:?} for cast {}", node.name)
                    }
                };
                let output = input.to_dtype(dtype)?;
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#CumSum
            "CumSum" => {
                let exclusive = parser::get_attr_opt::<i64>(node, "exclusive")?
                    .copied()
                    .unwrap_or(0);
                let reverse = parser::get_attr_opt::<i64>(node, "reverse")?
                    .copied()
                    .unwrap_or(0);
                if exclusive != 0 {
                    bail!("only exclusive == 0 is supported in CumSum")
                }
                if reverse != 0 {
                    bail!("only reverse == 0 is supported in CumSum")
                }
                let input = get(&node.input[0])?;
                let axis = get(&node.input[1])?
                    .to_dtype(DType::U32)?
                    .to_vec0::<u32>()?;
                let output = input.cumsum(axis as usize)?;
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#identity
            "Identity" => {
                let input = get(&node.input[0])?;
                values.insert(node.output[0].clone(), input.clone());
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#if
            "If" => {
                // protobuf encodes boolean false as 0 and true as 1
                let cond = get(&node.input[0])?.get(0)?.to_scalar::<u8>()?;
                let attr_name = if cond != 0 {
                    "then_branch"
                } else {
                    "else_branch"
                };
                let sub_graph = parser::get_attr::<GraphProto>(node, attr_name)?;
                if sub_graph.output.len() != node.output.len() {
                    bail!(
                        "If node {:?} is malformed: branch outputs ({}) don't match node outputs ({})",
                        node.name,
                        sub_graph.output.len(),
                        node.output.len()
                    );
                }
                let branch_out = simple_eval_(sub_graph, values)?;
                for (i, out) in node.output.iter().enumerate() {
                    values.insert(
                        out.clone(),
                        branch_out.get(&sub_graph.output[i].name).unwrap().clone(),
                    );
                }
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#pad
            "Pad" => {
                let mode = parser::get_attr_opt(node, "mode")?.unwrap_or("constant");
                let data = get(&node.input[0])?;
                let pads = get(&node.input[1])?;
                if node.input.len() > 2 {
                    bail!(
                        "unsupported number of inputs {} for Pad node {:?}, expected 2",
                        node.input.len(),
                        node.name
                    );
                }
                if pads.rank() != 1 {
                    bail!("Pad expects 'pads' input to be 1D vector: {pads:?}");
                }
                if pads.dim(0).unwrap() != 2 * data.rank() {
                    bail!("Pad expects 'pads' input len to be 2 * rank of 'data' input: pads: {}, data rank: {}", pads, data.rank());
                }

                let pads = pads.to_vec1::<i64>()?;
                let (pads_pre, pads_post) = pads.split_at(pads.len() / 2);

                match mode {
                    "reflect" => {
                        let mut out = data.clone();
                        for (i, &dim) in data.dims().iter().enumerate().rev() {
                            if pads_pre[i] == 0 && pads_post[i] == 0 {
                                continue;
                            }
                            fn zigzag(min: i64, max: i64) -> impl Iterator<Item = i64> {
                                std::iter::repeat((min..max).chain((min + 1..=max).rev())).flatten()
                            }
                            let idx = if dim > 1 {
                                let cycle_len = dim * 2 - 2;
                                let skip = cycle_len - ((pads_pre[i] as usize) % cycle_len);
                                let idx = zigzag(0, (dim - 1) as i64)
                                    .skip(skip)
                                    .take((pads_pre[i] as usize) + dim + (pads_post[i] as usize));
                                Tensor::from_iter(idx, out.device())?
                            } else {
                                Tensor::full(0i64, (dim,), out.device())?
                            };

                            out = out.index_select(&idx, i)?;
                        }

                        values.insert(node.output[0].clone(), out);
                    }
                    _ => bail!(
                        "unsupported 'mode' value {mode:?} for Pad node {:?}",
                        node.name
                    ),
                }
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#slice
            "Slice" => {
                let data = get(&node.input[0])?;
                let starts = get(&node.input[1])?;
                let ends = get(&node.input[2])?;
                let default_axes;
                let default_steps;
                let axes: &Tensor;
                let steps: &Tensor;
                // If axes are omitted, they are set to [0, ..., r-1]. If steps are omitted,
                // they are set to [1, ..., 1] of length len(starts)
                match node.input.len() {
                    3 => {
                        let len = starts.dims()[0];
                        default_axes = Some(Tensor::arange(0, len as i64, starts.device())?);
                        axes = default_axes.as_ref().unwrap();
                        default_steps = Some(Tensor::ones((len,), DType::I64, starts.device())?);
                        steps = default_steps.as_ref().unwrap();
                    }
                    4 => {
                        let len = starts.dims()[0];
                        axes = get(&node.input[3])?;
                        default_steps = Some(Tensor::ones((len,), DType::I64, starts.device())?);
                        steps = default_steps.as_ref().unwrap();
                    }
                    5 => {
                        steps = get(&node.input[4])?;
                        axes = get(&node.input[3])?;
                    }
                    _ => bail!(
                        "Slice node is invalid, expected 3-5 inputs, got {}: {:?}",
                        node.input.len(),
                        node
                    ),
                }

                let mut out = data.clone();
                for (i, axis) in axes.to_vec1::<i64>()?.into_iter().enumerate() {
                    // All negative elements of axes are made non-negative by
                    // adding r to them, where r = rank(input).
                    let axis = if axis < 0 {
                        axis + data.rank() as i64
                    } else {
                        axis
                    } as usize;

                    let data_dim = data.dims()[axis] as i64;
                    let mut s = starts.get(i)?.to_scalar::<i64>()?;
                    let mut e = ends.get(i)?.to_scalar::<i64>()?;
                    // All negative values in starts[i] and ends[i] have
                    // dims[axes[i]] added to them, where dims are the
                    // dimensions of input.
                    if s < 0 {
                        s += data_dim;
                    }
                    if e < 0 {
                        e += data_dim;
                    }

                    let p = steps.get(i)?.to_scalar::<i64>()?;
                    // starts[i] is clamped into the range [0, dims[axes[i]]]
                    // for positive stepping and [0, dims[axes[i]]-1] for
                    // negative stepping.
                    // for positive stepping ends[axes[i]] is clamped to
                    // [0, dims[axes[i]]], while for negative stepping it is
                    // clamped to [-1, dims[axes[i]]-1].
                    if p >= 0 {
                        s = s.clamp(0, data_dim);
                        e = e.clamp(0, data_dim);
                    } else {
                        s = s.clamp(0, data_dim - 1);
                        e = e.clamp(-1, data_dim - 1);
                    }

                    let indexes = Tensor::arange_step(s, e, p, data.device())?;
                    out = out.index_select(&indexes, axis)?
                }
                values.insert(node.output[0].clone(), out);
            }
            // https://onnx.ai/onnx/operators/onnx__ReduceMax.html#reducemax
            "ReduceMax" => {
                let input = get(&node.input[0])?;
                let axes = get_opt(1);
                let keepdims = parser::get_attr_opt::<i64>(node, "keepdims")?
                    .copied()
                    .unwrap_or(1)
                    == 1;

                let axes = if let Some(Ok(axes)) = axes {
                    // Satisfies version 18+
                    axes.to_vec1::<i64>().ok()
                } else if let Ok(Some(axes)) = parser::get_attr_opt::<[i64]>(node, "axes") {
                    // Backward compatiblity with version 13 and below
                    Some(axes.to_vec())
                } else {
                    None
                };

                let axes = if let Some(axes) = axes {
                    let rank = input.rank();
                    let mut axes_set = HashSet::new();

                    let mut axes = axes
                        .iter()
                        .map(|a| {
                            let axis = if *a < 0 {
                                (rank as i64 + *a) as usize
                            } else {
                                *a as usize
                            };

                            axes_set.insert(axis);
                            axis
                        })
                        .collect::<Vec<_>>();

                    if axes_set.len() < axes.len() {
                        bail!("Duplicate value in 'axes'");
                    }

                    if axes.len() > 1 {
                        axes.sort();
                    }

                    Some(axes)
                } else {
                    None
                };

                // TODO: Handle empty set
                // Definition:
                // "Reduction over an empty set of values yields minus infinity (if supported by the datatype) or the minimum value of the data type otherwise"
                // For now, this will throw an error
                if input.elem_count() == 0 {
                    bail!("reduction over zero-size tensor not supported");
                }

                let output = if let Some(axes) = axes {
                    let mut result = input.clone();
                    for &axis in axes.iter().rev() {
                        result = if keepdims {
                            result.max_keepdim(axis)?
                        } else {
                            result.max(axis)?
                        }
                    }

                    result
                } else {
                    // If `axes` is empty and `noop_with_empty_axes` is set to `true (1)`
                    // ""input tensor will not be reduced,and the output tensor would be equivalent to input tensor.""
                    if parser::get_attr_opt::<i64>(node, "noop_with_empty_axes")?.copied()
                        == Some(1)
                    {
                        input.clone()
                    } else {
                        let mut result = input.flatten_all()?;
                        if keepdims {
                            result = result.max_keepdim(0)?;
                            // If keepdims is true, reshape to match input dimensions
                            let shape = vec![1; input.rank()];
                            result.reshape(shape)?
                        } else {
                            result.max(0)?
                        }
                    }
                };

                values.insert(node.output[0].clone(), output);
            }
            // https://onnx.ai/onnx/operators/onnx__ReduceMean.html#reducemean-13
            // TODO: This version is only compatible with ReduceMean V13 and below.
            "ReduceMean" => {
                let input = get(&node.input[0])?;
                let axes = parser::get_attr_opt::<[i64]>(node, "axes")?;
                let keepdims = parser::get_attr_opt::<i64>(node, "keepdims")?
                    .copied()
                    .unwrap_or(1);

                let n_dims = input.dims().len();

                let axes: Vec<usize> = if let Some(axes) = axes {
                    axes.iter()
                        .map(|e| (if e < &0 { (n_dims as i64) + *e } else { *e }) as usize)
                        .collect()
                } else {
                    (0..n_dims).collect()
                };
                let output = if keepdims == 1 {
                    input.mean_keepdim(axes)?
                } else {
                    input.mean(axes)?
                };
                values.insert(node.output[0].clone(), output);
            }
            // https://onnx.ai/onnx/operators/onnx__ReduceMin.html#reducemin
            "ReduceMin" => {
                let input = get(&node.input[0])?;
                let axes = get_opt(1);
                let keepdims = parser::get_attr_opt::<i64>(node, "keepdims")?
                    .copied()
                    .unwrap_or(1)
                    == 1;

                let axes = if let Some(Ok(axes)) = axes {
                    // Satisfies version 18+
                    axes.to_vec1::<i64>().ok()
                } else if let Ok(Some(axes)) = parser::get_attr_opt::<[i64]>(node, "axes") {
                    // Backward compatiblity with version 13 and below
                    Some(axes.to_vec())
                } else {
                    None
                };

                let axes = if let Some(axes) = axes {
                    let rank = input.rank();
                    let mut axes_set = HashSet::new();

                    let mut axes = axes
                        .iter()
                        .map(|a| {
                            let axis = if *a < 0 {
                                (rank as i64 + *a) as usize
                            } else {
                                *a as usize
                            };

                            axes_set.insert(axis);
                            axis
                        })
                        .collect::<Vec<_>>();

                    if axes_set.len() < axes.len() {
                        bail!("Duplicate value in 'axes'");
                    }

                    if axes.len() > 1 {
                        axes.sort();
                    }

                    Some(axes)
                } else {
                    None
                };

                // TODO: Handle empty set
                // Definition:
                // "Reduction over an empty set of values yields positive infinity (if supported by the datatype) or the max value of the data type otherwise"
                // For now, this will throw an error
                if input.elem_count() == 0 {
                    bail!("reduction over zero-size tensor not supported");
                }

                let output = if let Some(axes) = axes {
                    let mut result = input.clone();
                    for &axis in axes.iter().rev() {
                        result = if keepdims {
                            result.min_keepdim(axis)?
                        } else {
                            result.min(axis)?
                        }
                    }

                    result
                } else {
                    // If `axes` is empty and `noop_with_empty_axes` is set to `true (1)`
                    // ""input tensor will not be reduced,and the output tensor would be equivalent to input tensor.""
                    if parser::get_attr_opt::<i64>(node, "noop_with_empty_axes")?.copied()
                        == Some(1)
                    {
                        input.clone()
                    } else {
                        let mut result = input.flatten_all()?;
                        if keepdims {
                            result = result.min_keepdim(0)?;
                            // If keepdims is true, reshape to match input dimensions
                            let shape = vec![1; input.rank()];
                            result.reshape(shape)?
                        } else {
                            result.min(0)?
                        }
                    }
                };

                values.insert(node.output[0].clone(), output);
            }
            //https://github.com/onnx/onnx/blob/main/docs/Operators.md#Split
            // Version 18 impl
            "Split" => {
                let input_tensor = get(&node.input[0])?;
                let axis = parser::get_attr_opt::<i64>(node, "axis")?
                    .copied()
                    .unwrap_or(0);
                let axis = input_tensor.normalize_axis(axis)?;

                // Determine split sizes
                let splits = if node.input.len() > 1 {
                    // If the split tensor is provided, use it to determine sizes
                    let split_tensor = get(&node.input[1])?.to_vec1::<i64>()?;
                    split_tensor.iter().map(|&x| x as usize).collect::<Vec<_>>()
                } else {
                    let num_outputs = if let Some(&num_outputs_attrib) =
                        parser::get_attr_opt::<i64>(node, "num_outputs")?
                    {
                        num_outputs_attrib as usize
                    } else {
                        node.output.len()
                    };

                    let input_dim = input_tensor.dim(axis)?;

                    let mut split_sizes =
                        vec![input_dim / num_outputs as usize; num_outputs as usize];
                    let remainder = input_dim % num_outputs as usize;
                    if remainder > 0 {
                        // If there's a remainder, add it to the last split size
                        split_sizes[num_outputs as usize - 1] += remainder;
                    }

                    split_sizes
                };

                // Perform the split operation
                let mut outputs = vec![];
                let mut start = 0;
                for &size in &splits {
                    let end = start + size;
                    let slice = input_tensor.narrow(axis, start, size)?;
                    outputs.push(slice);
                    start = end;
                }

                // Insert the split outputs into the values map
                for (output, slice) in node.output.iter().zip(outputs.into_iter()) {
                    values.insert(output.clone(), slice);
                }
            }
            //https://github.com/onnx/onnx/blob/main/docs/Operators.md#Expand
            // Version 13 impl
            "Expand" => {
                // unlike broadcast_to, expand allows for the output shape to
                // be different from the specified shape.
                let input_tensor = get(&node.input[0])?;
                let input_shape = get(&node.input[1])?;

                // Check that the shape tensor is 1D
                if input_shape.rank() != 1 {
                    bail!(
                        "Expand expects 'shape' input to be 1D tensor: {:?}",
                        input_shape
                    );
                }
                let input_tensor_dims = input_tensor.dims();
                let input_shape_dims = input_shape
                    .to_vec1::<i64>()?
                    .into_iter()
                    .map(|x| x as usize)
                    .collect::<Vec<_>>();

                let target_shape = broadcast_shape(input_tensor_dims, input_shape_dims.as_slice())?;

                let expanded_tensor = input_tensor.broadcast_as(target_shape)?;

                values.insert(node.output[0].clone(), expanded_tensor);
            }
            //https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSum
            // Version 13 impl
            "ReduceSum" => {
                let input = get(&node.input[0])?;
                let axes = get_opt(1);
                let keepdims = parser::get_attr_opt::<i64>(node, "keepdims")?
                    .copied()
                    .unwrap_or(1);
                let noop_with_empty_axes =
                    parser::get_attr_opt::<i64>(node, "noop_with_empty_axes")?
                        .copied()
                        .unwrap_or(0);

                let axes = match axes {
                    Some(Ok(axes)) => axes
                        .to_vec1::<i64>()?
                        .into_iter()
                        .map(|x| x as usize)
                        .collect::<Vec<_>>(),
                    Some(Err(_)) | None => {
                        if noop_with_empty_axes == 1 {
                            vec![]
                        } else {
                            (0..input.rank()).collect()
                        }
                    }
                };

                let output = if keepdims == 1 {
                    input.sum_keepdim(axes)?
                } else {
                    input.sum(axes)?
                };

                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceL2
            // Version 18 impl
            "ReduceL2" => {
                let input = get(&node.input[0])?;
                let axes = get_opt(1);
                let keepdims = parser::get_attr_opt::<i64>(node, "keepdims")?
                    .copied()
                    .unwrap_or(1);
                let noop_with_empty_axes =
                    parser::get_attr_opt::<i64>(node, "noop_with_empty_axes")?
                        .copied()
                        .unwrap_or(0);

                let input_sq = input.sqr()?;

                let axes = match axes {
                    Some(axes) => axes?
                        .to_vec1::<i64>()?
                        .into_iter()
                        .map(|x| x as usize)
                        .collect::<Vec<_>>(),
                    None => {
                        if noop_with_empty_axes == 1 {
                            vec![]
                        } else {
                            (0..input_sq.rank()).collect()
                        }
                    }
                };

                let output = if keepdims == 1 {
                    input_sq.sum_keepdim(axes)?.sqrt()?
                } else {
                    input_sq.sum(axes)?.sqrt()?
                };

                values.insert(node.output[0].clone(), output);
            }
            random_type @ ("RandomUniform" | "RandomNormal") => {
                let dt: i64 = parser::get_attr_opt(node, "dtype")?.copied().unwrap_or(1); // 1 is float
                                                                                          // type by
                                                                                          // default
                let dtype = match DataType::try_from(dt as i32) {
                    Ok(dt) => match parser::dtype(dt) {
                        Some(DType::U8 | DType::U32 | DType::I64) => {
                            bail!(
                                "unsupported 'dtype' value {dt:?}, only floats are allowed, for {random_type} {}",
                                node.name
                            )
                        }
                        Some(dt) => dt,
                        None => {
                            bail!(
                                "unsupported 'dtype' value {dt:?} for {random_type} {}",
                                node.name
                            )
                        }
                    },
                    Err(_) => {
                        bail!(
                            "unsupported 'dtype' value {dt:?} for {random_type} {}",
                            node.name
                        )
                    }
                };
                let seed: Option<f32> = parser::get_attr_opt(node, "seed")?.copied();
                if seed.is_some() {
                    bail!("seed for {random_type} is currently not supported")
                };
                let shape: Vec<usize> = parser::get_attr::<[i64]>(node, "shape")?
                    .iter()
                    .map(|x| *x as usize)
                    .collect();
                let output = if random_type == "RandomUniform" {
                    let low: f32 = parser::get_attr_opt(node, "low")?.copied().unwrap_or(0.0);
                    let high: f32 = parser::get_attr_opt(node, "high")?.copied().unwrap_or(1.0);
                    Tensor::rand(low, high, shape, &Device::Cpu)?.to_dtype(dtype)?
                } else {
                    let mean: f32 = parser::get_attr_opt(node, "mean")?.copied().unwrap_or(0.0);
                    let scale: f32 = parser::get_attr_opt(node, "scale")?.copied().unwrap_or(1.0);
                    Tensor::randn(mean, scale, shape, &Device::Cpu)?.to_dtype(dtype)?
                };
                values.insert(node.output[0].clone(), output);
            }
            "ArgMin" => {
                let input = get(&node.input[0])?;
                let axis_i64: i64 = parser::get_attr_opt(node, "axis")?.copied().unwrap_or(0);
                let rank_i64: i64 = input.rank().try_into().unwrap();
                if axis_i64 < -rank_i64 || axis_i64 >= rank_i64 {
                    bail!(
                        "axis ({}) out of accepted range [-rank, rank-1] which was [{}, {}]",
                        axis_i64,
                        -rank_i64,
                        rank_i64 - 1
                    )
                }
                let axis = input.normalize_axis(axis_i64)?;
                let keepdims: i64 = parser::get_attr_opt(node, "keepdims")?
                    .copied()
                    .unwrap_or(1);
                let select_last_index: i64 = parser::get_attr_opt(node, "select_last_index")?
                    .copied()
                    .unwrap_or(0);
                if select_last_index == 1 {
                    bail!("select_last_index for ArgMin is currently not supported")
                }
                let output = if keepdims == 1 {
                    input.argmin_keepdim(axis)?
                } else {
                    input.argmin(axis)?
                }
                .to_dtype(DType::I64)?;
                values.insert(node.output[0].clone(), output);
            }
            "ArgMax" => {
                let input = get(&node.input[0])?;
                let axis_i64: i64 = parser::get_attr_opt(node, "axis")?.copied().unwrap_or(0);
                let rank_i64: i64 = input.rank().try_into().unwrap();
                if axis_i64 < -rank_i64 || axis_i64 >= rank_i64 {
                    bail!(
                        "axis ({}) out of accepted range [-rank, rank-1] which was [{}, {}]",
                        axis_i64,
                        -rank_i64,
                        rank_i64 - 1
                    )
                }
                let axis = input.normalize_axis(axis_i64)?;
                let keepdims: i64 = parser::get_attr_opt(node, "keepdims")?
                    .copied()
                    .unwrap_or(1);
                let select_last_index: i64 = parser::get_attr_opt(node, "select_last_index")?
                    .copied()
                    .unwrap_or(0);
                if select_last_index == 1 {
                    bail!("select_last_index for ArgMin is currently not supported")
                }
                let output = if keepdims == 1 {
                    input.argmax_keepdim(axis)?
                } else {
                    input.argmax(axis)?
                }
                .to_dtype(DType::I64)?;
                values.insert(node.output[0].clone(), output);
            }
            "LeakyRelu" => {
                let input = get(&node.input[0])?;
                let dt = input.dtype();
                match dt {
                    DType::U8 | DType::U32 | DType::I64 => {
                        bail!(
                            "unsupported dtype {}, only float types are allowed for LeakyRelu",
                            dt.as_str()
                        )
                    }
                    DType::BF16 | DType::F16 | DType::F32 | DType::F64 => {}
                }
                let alpha = parser::get_attr_opt::<f32>(node, "alpha")?
                    .copied()
                    .unwrap_or(0.01);
                let output = candle_nn::ops::leaky_relu(input, alpha.into())?;
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm
            "Gemm" => {
                let a = get(&node.input[0])?;
                let b = get(&node.input[1])?;
                let c = get(&node.input[2])?;

                let alpha = parser::get_attr_opt::<f32>(node, "alpha")?
                    .copied()
                    .unwrap_or(1.0);
                let beta = parser::get_attr_opt::<f32>(node, "beta")?
                    .copied()
                    .unwrap_or(1.0);

                let alpha = Tensor::full(alpha, a.shape(), &Device::Cpu)?;
                let beta = Tensor::full(beta, c.shape(), &Device::Cpu)?;

                let trans_a = parser::get_attr_opt::<i64>(node, "transA")?
                    .copied()
                    .unwrap_or(0);
                let trans_b = parser::get_attr_opt::<i64>(node, "transB")?
                    .copied()
                    .unwrap_or(0);

                let a = if trans_a == 0 { a.clone() } else { a.t()? };
                let b = if trans_b == 0 { b.clone() } else { b.t()? };

                let output = a
                    .broadcast_mul(&alpha)?
                    .broadcast_matmul(&b)?
                    .broadcast_add(&c.broadcast_mul(&beta)?)?;
                values.insert(node.output[0].clone(), output);
            }
            "LSTM" => {
                let direction = parser::get_attr_opt(node, "direction")?.unwrap_or("forward");
                if direction != "forward" {
                    bail!("LSTM currently only supports direction == \"forward\"");
                }
                let num_directions = if direction == "bidirectional" { 2 } else { 1 };
                let hidden_size: i64 = parser::get_attr(node, "hidden_size").copied()?;
                let input_forget = parser::get_attr_opt(node, "input_forget")?
                    .copied()
                    .unwrap_or(0);
                if input_forget != 0 {
                    bail!("LSTM currently only supports input_forget == 0");
                }
                let activations_default = vec![
                    "Sigmoid".to_string(),
                    "Tanh".to_string(),
                    "Tanh".to_string(),
                ];
                let activations = parser::get_attr_opt_owned::<Vec<String>>(node, "activations")?
                    .unwrap_or(activations_default.clone());
                if activations != activations_default {
                    bail!("LSTM currently only supports default activations ({activations_default:?})");
                }
                // activation_alpha and activation_beta don't apply to (Sigmoid, Tanh, Tanh) so ignoring them is okay
                if parser::get_attr_opt::<f32>(node, "clip")?.is_some() {
                    bail!("LSTM does not currently support clip attribute");
                }

                // The shape format of inputs X, initial_h and outputs Y, Y_h.
                // If 0, the following shapes are expected:
                //     X.shape = [seq_length, batch_size, input_size],
                //     Y.shape = [seq_length, num_directions, batch_size, hidden_size],
                //     initial_h.shape = Y_h.shape = [num_directions, batch_size, hidden_size].
                // If 1, the following shapes are expected:
                //     X.shape = [batch_size, seq_length, input_size],
                //     Y.shape = [batch_size, seq_length, num_directions, hidden_size],
                //     initial_h.shape = Y_h.shape = [batch_size, num_directions, hidden_size].
                let layout = parser::get_attr_opt(node, "layout")?.copied().unwrap_or(0);
                if layout != 0 {
                    bail!("LSTM currently only supports layout == 0");
                }

                // The input sequences packed (and potentially padded) into one 3-D tensor
                // with the shape of `[seq_length, batch_size, input_size]`.
                let x = get(&node.input[0])?;
                // XXX: depends on layout
                let (seq_length, batch_size, input_size) = x.dims3()?;
                // The weight tensor for the gates.
                // Concatenation of `W[iofc]` and `WB[iofc]` (if bidirectional) along dimension 0.
                // The tensor has shape `[num_directions, 4*hidden_size, input_size]`.
                let w = get(&node.input[1])?;
                // The recurrence weight tensor.
                // Concatenation of `R[iofc]` and `RB[iofc]` (if bidirectional) along dimension 0.
                // This tensor has shape `[num_directions, 4*hidden_size, hidden_size]`.
                let r = get(&node.input[2])?;

                // The bias tensor for input gate.
                // Concatenation of `[Wb[iofc], Rb[iofc]]`, and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along dimension 0.
                // This tensor has shape `[num_directions, 8*hidden_size]`.
                // Optional: If not specified - assumed to be 0.
                let b_default: Tensor;
                let b = match get_opt(3) {
                    Some(n) => n?,
                    None => {
                        b_default = Tensor::zeros(
                            (num_directions, 8 * hidden_size as usize),
                            DType::F32,
                            x.device(),
                        )?;
                        &b_default
                    }
                };

                // Optional tensor specifying lengths of the sequences in a batch.
                // If not specified - assumed all sequences in the batch to have length `seq_length`.
                // It has shape `[batch_size]`.
                let seq_lens_default: Tensor;
                let seq_lens = match get_opt(4) {
                    Some(n) => n?,
                    None => {
                        seq_lens_default =
                            Tensor::full(seq_length as i64, (batch_size,), x.device())?;
                        &seq_lens_default
                    }
                };
                let seq_lens_is_default =
                    (seq_lens.to_vec1::<i64>()?.iter()).all(|e| *e as usize == seq_length);
                if !seq_lens_is_default {
                    bail!("LSTM currently only supports default value of seq_lens");
                }

                // Optional initial value of the hidden. If not specified - assumed to be 0.
                // It has shape `[num_directions, batch_size, hidden_size]`.
                let initial_h_default: Tensor;
                let initial_h = match get_opt(5) {
                    Some(n) => n?,
                    _ => {
                        initial_h_default = Tensor::zeros(
                            (num_directions, batch_size, hidden_size as usize),
                            DType::F32,
                            x.device(),
                        )?;
                        &initial_h_default
                    }
                };

                // Optional initial value of the cell.
                // If not specified - assumed to be 0.
                // It has shape `[num_directions, batch_size, hidden_size]`.
                let initial_c_default: Tensor;
                let initial_c = match node.input.get(6) {
                    Some(n) if !n.is_empty() => get(n)?,
                    _ => {
                        initial_c_default = Tensor::zeros(
                            (num_directions, batch_size, hidden_size as usize),
                            DType::F32,
                            x.device(),
                        )?;
                        &initial_c_default
                    }
                };

                // The weight tensor for peepholes.
                // Concatenation of `P[iof]` and `PB[iof]` (if bidirectional) along dimension 0.
                // It has shape `[num_directions, 3*hidde_size]`. Optional: If not specified - assumed to be 0.
                let p_default = Tensor::zeros(
                    (num_directions, 3 * hidden_size as usize),
                    DType::F32,
                    x.device(),
                )?;
                let p = get_opt(7).unwrap_or(Ok(&p_default))?;
                let p_is_zeros = (p.to_vec2::<f32>()?.iter()).all(|v| v.iter().all(|e| *e == 0.0));
                if !p_is_zeros {
                    bail!(
                        "LSTM currently only supports default value of p (a Tensor of all zeroes)"
                    );
                }

                // these all have [num_directions, ...] shapes
                let w = w.get(0)?; // w[iofc] has shape [4*hidden_size, input_size]
                let r = r.get(0)?; // r[iofc] has shape [4*hidden_size, hidden_size]
                let b = b.get(0)?; // concat of [wb[iofc],rb[iofc]] has shape [8*hidden_size]
                let idx_wb = Tensor::arange(0, 4 * hidden_size, x.device())?;
                let idx_rb = Tensor::arange(4 * hidden_size, 8 * hidden_size, x.device())?;
                let wb = b.index_select(&idx_wb, 0)?;
                let rb = b.index_select(&idx_rb, 0)?;
                let c = initial_c.get(0)?;
                let h = initial_h.get(0)?;

                // w, r, wb, rb are all iofc but lstm expects ifco
                // so we need to move some stuff around
                let idx_i = Tensor::arange(0, hidden_size, x.device())?;
                let idx_o = Tensor::arange(hidden_size, 2 * hidden_size, x.device())?;
                let idx_f = Tensor::arange(2 * hidden_size, 3 * hidden_size, x.device())?;
                let idx_c = Tensor::arange(3 * hidden_size, 4 * hidden_size, x.device())?;
                let idx_ifco = Tensor::cat(&[&idx_i, &idx_f, &idx_c, &idx_o], 0)?;
                let w = w.index_select(&idx_ifco, 0)?;
                let r = r.index_select(&idx_ifco, 0)?;
                let wb = wb.index_select(&idx_ifco, 0)?;
                let rb = rb.index_select(&idx_ifco, 0)?;
                let vmap = candle_nn::VarMap::new();
                vmap.data().lock().unwrap().extend([
                    ("weight_ih_l0".to_string(), candle::Var::from_tensor(&w)?),
                    ("weight_hh_l0".to_string(), candle::Var::from_tensor(&r)?),
                    ("bias_ih_l0".to_string(), candle::Var::from_tensor(&wb)?),
                    ("bias_hh_l0".to_string(), candle::Var::from_tensor(&rb)?),
                ]);
                use crate::parser;
                use candle_nn::rnn::RNN as _;
                let lstm = candle_nn::rnn::lstm(
                    input_size,
                    hidden_size as usize,
                    candle_nn::rnn::LSTMConfig::default(),
                    candle_nn::VarBuilder::from_varmap(&vmap, w.dtype(), w.device()),
                )?;

                let mut lstm_state = candle_nn::rnn::LSTMState::new(h, c);
                let mut h_acc = if node.output.first().map(String::as_str).unwrap_or("") != "" {
                    Some(vec![])
                } else {
                    None
                };
                for t in 0..seq_length {
                    let x = x.get(t)?;
                    lstm_state = lstm.step(&x, &lstm_state)?;
                    if let Some(h_acc) = &mut h_acc {
                        h_acc.push(lstm_state.clone());
                    }
                }

                assert_eq!(num_directions, 1, "if support for bidirectional is ever added, outputs will have to be concatenated, not simply reshaped");
                if let Some(name) = node.output.first() {
                    let h_acc = h_acc.as_ref().unwrap();
                    let h_acc = lstm.states_to_tensor(h_acc)?;
                    let h_acc = h_acc.reshape((
                        seq_length,
                        num_directions,
                        batch_size,
                        hidden_size as usize,
                    ))?;
                    values.insert(name.clone(), h_acc);
                }
                if let Some(name) = node.output.get(1) {
                    values.insert(
                        name.clone(),
                        lstm_state.h().reshape((
                            num_directions,
                            batch_size,
                            hidden_size as usize,
                        ))?,
                    );
                }
                if let Some(name) = node.output.get(2) {
                    values.insert(
                        name.clone(),
                        lstm_state.c().reshape((
                            num_directions,
                            batch_size,
                            hidden_size as usize,
                        ))?,
                    );
                }
            }
            op_type => {
                let onnx_op = registry.get(op_type)?;
                let cn = ComputeNode::new(&node, values);
                let (name, value) = onnx_op.eval(&cn)?;
                values.insert(name, value);
            }
        }
    }
    graph
        .output
        .iter()
        .map(|output| match values.remove(&output.name) {
            None => bail!("cannot find output {}", output.name),
            Some(value) => Ok((output.name.clone(), value)),
        })
        .collect()
}

fn broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> Result<Vec<usize>> {
    let (longest, shortest) = if shape_a.len() > shape_b.len() {
        (shape_a, shape_b)
    } else {
        (shape_b, shape_a)
    };
    let diff = longest.len() - shortest.len();
    let mut target_shape = longest[0..diff].to_vec();
    for (dim1, dim2) in longest[diff..].iter().zip(shortest.iter()) {
        if *dim1 == *dim2 || *dim2 == 1 || *dim1 == 1 {
            target_shape.push(usize::max(*dim1, *dim2));
        } else {
            bail!(
                "Expand: incompatible shapes for broadcast, {:?} and {:?}",
                shape_a,
                shape_b
            );
        }
    }
    Ok(target_shape)
}
