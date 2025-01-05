use crate::ops::OnnxOpError::InvalidInput;
use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};
use candle_core::{DType, Tensor};

pub(crate) struct Lstm;
impl OnnxOp for Lstm {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let direction = node.get_attr_opt("direction")?.unwrap_or("forward");
        if direction != "forward" {
            return Err(OnnxOpError::UnsupportedAttribute("LSTM currently only supports direction == \"forward\"".to_string()))
        }
        let num_directions = if direction == "bidirectional" { 2 } else { 1 };
        let hidden_size: i64 = node.get_attr("hidden_size").copied()?;
        let input_forget = node.get_attr_opt("input_forget")?
            .copied()
            .unwrap_or(0);
        if input_forget != 0 {
            return Err(OnnxOpError::UnsupportedAttribute("LSTM currently only supports input_forget == 0".to_string()))
        }
        let activations_default = vec![
            "Sigmoid".to_string(),
            "Tanh".to_string(),
            "Tanh".to_string(),
        ];
        let activations = node.get_attr_opt_owned::<Vec<String>>("activations")?
            .unwrap_or(activations_default.clone());
        if activations != activations_default {
            let msg = format!("LSTM currently only supports default activations ({activations_default:?})");
            return Err(OnnxOpError::UnsupportedAttribute(msg))

        }
        // activation_alpha and activation_beta don't apply to (Sigmoid, Tanh, Tanh) so ignoring them is okay
        if node.get_attr_opt::<f32>("clip")?.is_some() {
            return Err(OnnxOpError::UnsupportedAttribute("LSTM does not currently support clip attribute".to_string()))
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
        let layout = node.get_attr_opt("layout")?.copied().unwrap_or(0);
        if layout != 0 {
            return Err(OnnxOpError::UnsupportedAttribute("LSTM currently only supports layout == 0".to_string()));
        }

        // The input sequences packed (and potentially padded) into one 3-D tensor
        // with the shape of `[seq_length, batch_size, input_size]`.
        let x = node.get_input(0)?;
        // XXX: depends on layout
        let (seq_length, batch_size, input_size) = x.dims3()?;
        // The weight tensor for the gates.
        // Concatenation of `W[iofc]` and `WB[iofc]` (if bidirectional) along dimension 0.
        // The tensor has shape `[num_directions, 4*hidden_size, input_size]`.
        let w = node.get_input(1)?;
        // The recurrence weight tensor.
        // Concatenation of `R[iofc]` and `RB[iofc]` (if bidirectional) along dimension 0.
        // This tensor has shape `[num_directions, 4*hidden_size, hidden_size]`.
        let r = node.get_input(2)?;

        // The bias tensor for input gate.
        // Concatenation of `[Wb[iofc], Rb[iofc]]`, and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along dimension 0.
        // This tensor has shape `[num_directions, 8*hidden_size]`.
        // Optional: If not specified - assumed to be 0.
        let b_default: Tensor;
        let b = match node.get_opt(3) {
            Some(n) => n,
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
        let seq_lens = match node.get_opt(4) {
            Some(n) => n,
            None => {
                seq_lens_default =
                    Tensor::full(seq_length as i64, (batch_size,), x.device())?;
                &seq_lens_default
            }
        };
        let seq_lens_is_default =
            (seq_lens.to_vec1::<i64>()?.iter()).all(|e| *e as usize == seq_length);
        if !seq_lens_is_default {
            return Err(InvalidInput("LSTM currently only supports default value of seq_lens".to_string()));
        }

        // Optional initial value of the hidden. If not specified - assumed to be 0.
        // It has shape `[num_directions, batch_size, hidden_size]`.
        let initial_h_default: Tensor;
        let initial_h = match node.get_opt(5) {
            Some(n) => n,
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
        let initial_c_default = Tensor::zeros(
            (num_directions, batch_size, hidden_size as usize),
            DType::F32,
            x.device(),
        )?;
        let initial_c = node.get_opt(6).unwrap_or(&initial_c_default);

        // The weight tensor for peepholes.
        // Concatenation of `P[iof]` and `PB[iof]` (if bidirectional) along dimension 0.
        // It has shape `[num_directions, 3*hidde_size]`. Optional: If not specified - assumed to be 0.
        let p_default = Tensor::zeros(
            (num_directions, 3 * hidden_size as usize),
            DType::F32,
            x.device(),
        )?;
        let p = node.get_opt(7).unwrap_or(&p_default);
        let p_is_zeros = (p.to_vec2::<f32>()?.iter()).all(|v| v.iter().all(|e| *e == 0.0));
        if !p_is_zeros {
            return Err(InvalidInput("LSTM currently only supports default value of p (a Tensor of all zeroes)".to_string()))
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
            ("weight_ih_l0".to_string(), candle_core::Var::from_tensor(&w)?),
            ("weight_hh_l0".to_string(), candle_core::Var::from_tensor(&r)?),
            ("bias_ih_l0".to_string(), candle_core::Var::from_tensor(&wb)?),
            ("bias_hh_l0".to_string(), candle_core::Var::from_tensor(&rb)?),
        ]);
        use candle_nn::rnn::RNN as _;
        let lstm = candle_nn::rnn::lstm(
            input_size,
            hidden_size as usize,
            candle_nn::rnn::LSTMConfig::default(),
            candle_nn::VarBuilder::from_varmap(&vmap, w.dtype(), w.device()),
        )?;

        let mut lstm_state = candle_nn::rnn::LSTMState::new(h, c);
        let mut h_acc = if node.get_output(0).is_ok() {
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
        let mut values:Vec<(String, Tensor)> = Vec::new();
        if let Some(name) = node.get_output_opt(0) {
            let h_acc = h_acc.as_ref().unwrap();
            let h_acc = lstm.states_to_tensor(h_acc)?;
            let h_acc = h_acc.reshape((
                seq_length,
                num_directions,
                batch_size,
                hidden_size as usize,
            ))?;
            values.push((name.clone(), h_acc));
        }
        if let Some(name) = node.get_output_opt(1) {
            values.push((
                name.clone(),
                lstm_state.h().reshape((
                    num_directions,
                    batch_size,
                    hidden_size as usize,
                ))?),
            );
        }
        if let Some(name) = node.get_output_opt(2) {
            values.push((
                name.clone(),
                lstm_state.c().reshape((
                    num_directions,
                    batch_size,
                    hidden_size as usize,
                ))?),
            );
        }

        Ok(OpOutput::Multiple(values))
    }
}