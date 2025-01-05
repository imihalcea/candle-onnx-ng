use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};

pub(crate) struct Split;

impl OnnxOp for Split {
    //https://github.com/onnx/onnx/blob/main/docs/Operators.md#Split
    // Version 18 impl
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let input_tensor = node.get_input(0)?;
        let axis = node.get_attr_opt::<i64>("axis")?.copied().unwrap_or(0);
        let axis = input_tensor.normalize_axis(axis)?;

        // Determine split sizes
        let splits = if node.input_len() > 1 {
            // If the split tensor is provided, use it to determine sizes
            let split_tensor = &node.get_input(1)?.to_vec1::<i64>()?;
            split_tensor.iter().map(|&x| x as usize).collect::<Vec<_>>()
        } else {
            let num_outputs =
                if let Some(&num_outputs_attrib) = node.get_attr_opt::<i64>("num_outputs")? {
                    num_outputs_attrib as usize
                } else {
                    node.output_len()
                };

            let input_dim = input_tensor.dim(axis)?;

            let mut split_sizes = vec![input_dim / num_outputs; num_outputs];
            let remainder = input_dim % num_outputs;
            if remainder > 0 {
                // If there's a remainder, add it to the last split size
                split_sizes[num_outputs - 1] += remainder;
            }

            split_sizes
        };

        // Perform the split operation
        let mut slices = vec![];
        let mut start = 0;
        for &size in &splits {
            let end = start + size;
            let slice = input_tensor.narrow(axis, start, size)?;
            slices.push(slice);
            start = end;
        }

        let mut outputs = vec![];
        for (output_idx, slice) in (0..node.output_len()).zip(slices.into_iter()) {
            let output = node.get_output(output_idx)?;
            outputs.push((output.clone(), slice));
        }
        Ok(OpOutput::Multiple(outputs))
    }
}
