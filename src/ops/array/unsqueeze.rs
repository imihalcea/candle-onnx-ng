use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};
use candle_core::Error;

pub(crate) struct Unsqueeze;

impl OnnxOp for Unsqueeze {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let xs = node.get_input(0)?;
        let axes = match node.get_attr_opt::<[i64]>("axes")? {
            Some(axis) => axis.to_vec(),
            None => node.get_input(1)?.to_vec1::<i64>()?,
        };
        let mut axes = axes
            .iter()
            .map(|&i| {
                if i == xs.rank() as i64 {
                    Ok(xs.rank())
                } else if i < 0 {
                    // normalize_axis doesn't work correctly here
                    // because we actually want normalized with respect
                    // to the final size, not the current (off by one)
                    Ok(xs.rank() - (-i as usize) + 1)
                } else {
                    xs.normalize_axis(i)
                }
            })
            .collect::<Result<Vec<_>, Error>>()?;

        axes.sort();
        let mut xs = xs.clone();
        for &axis in axes.iter().rev() {
            xs = xs.unsqueeze(axis)?
        }
        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), xs))
    }
}
