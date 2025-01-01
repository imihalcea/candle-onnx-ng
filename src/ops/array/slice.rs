use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};
use candle_core::{DType, Tensor};

pub(crate) struct Slice;

impl OnnxOp for Slice {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let data = node.get_input(0)?;
        let starts = node.get_input(1)?;
        let ends = node.get_input(2)?;
        let default_axes;
        let default_steps;
        let axes: &Tensor;
        let steps: &Tensor;
        // If axes are omitted, they are set to [0, ..., r-1]. If steps are omitted,
        // they are set to [1, ..., 1] of length len(starts)
        match node.input_len() {
            3 => {
                let len = starts.dims()[0];
                default_axes = Some(Tensor::arange(0, len as i64, starts.device())?);
                axes = default_axes.as_ref().unwrap();
                default_steps = Some(Tensor::ones((len,), DType::I64, starts.device())?);
                steps = default_steps.as_ref().unwrap();
            }
            4 => {
                let len = starts.dims()[0];
                axes = node.get_input(3)?;
                default_steps = Some(Tensor::ones((len,), DType::I64, starts.device())?);
                steps = default_steps.as_ref().unwrap();
            }
            5 => {
                steps = node.get_input(4)?;
                axes = node.get_input(3)?;
            }
            _ => {
                let err_msg = format!(
                    "Slice node is invalid, expected 3-5 inputs, got {}: {:?}",
                    node.input_len(),
                    node.name
                );
                return Err(OnnxOpError::InvalidInput(err_msg));
            }
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
        let output_name = node.get_output(0)?;
        Ok(OpOutput::Single(output_name.clone(), out))
    }
}
