use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};

pub(crate) struct BatchNormalization;

impl OnnxOp for BatchNormalization {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let training_mode = node.get_attr_opt::<i64>("training_mode")?;
        if training_mode.copied().unwrap_or(0) != 0 {
            return Err(OnnxOpError::UnsupportedOp(
                "training mode is not supported for BatchNorm".to_string(),
            ));
        }
        let eps = node
            .get_attr_opt::<f32>("epsilon")?
            .copied()
            .unwrap_or(1e-5);
        let xs = node.get_input(0)?;
        let weight = node.get_input(1)?;
        let bias = node.get_input(2)?;
        let running_mean = node.get_input(3)?;
        let running_var = node.get_input(4)?;

        let target_shape: Vec<usize> = xs
            .dims()
            .iter()
            .enumerate()
            .map(|(idx, v)| if idx == 1 { *v } else { 1 })
            .collect();

        let target_shape = target_shape.as_slice();

        let xs = xs
            .broadcast_sub(&running_mean.reshape(target_shape)?)?
            .broadcast_div(&(running_var.reshape(target_shape)? + eps as f64)?.sqrt()?)?;

        let weight = weight.reshape(target_shape)?;
        let bias = bias.reshape(target_shape)?;
        let output = xs.broadcast_mul(&weight)?.broadcast_add(&bias)?;
        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}
