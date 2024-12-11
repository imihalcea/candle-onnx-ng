use std::collections::HashSet;
use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};

pub(crate) struct ReduceMin;
impl OnnxOp for ReduceMin {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        // https://onnx.ai/onnx/operators/onnx__ReduceMin.html#reducemin
        
        let input = node.get_input(0)?;
        let axes = node.get_opt(1);
        let keepdims = node.get_attr_opt::<i64>("keepdims")?
            .copied()
            .unwrap_or(1)
            == 1;

        let axes = if let Some(axes) = axes {
            // Satisfies version 18+
            axes.to_vec1::<i64>().ok()
        } else if let Ok(Some(axes)) = node.get_attr_opt::<[i64]>("axes") {
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
                return Err(OnnxOpError::InvalidInput(
                    "Duplicate value in 'axes'".to_string(),
                ));
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
            return Err(OnnxOpError::InvalidInput(
                "reduction over empty set not supported".to_string(),
            ));
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
            if node.get_attr_opt::<i64>("noop_with_empty_axes")?.copied()
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
        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}