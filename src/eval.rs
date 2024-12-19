use crate::onnx::tensor_proto::DataType;
use crate::onnx::{self, GraphProto};
use crate::ops::{registry, ComputeNode};
use candle_core as candle;
use candle_core::{bail, DType, Device, Result, Tensor};

use crate::parser;
use crate::parser::Value;
use std::collections::HashMap;

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
