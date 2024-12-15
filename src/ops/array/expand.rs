use crate::ops::tensor_helper::broadcast_shape;
use crate::ops::{ComputeNode, OnnxOp, OnnxOpError, OpOutput};

pub(crate) struct Expand;

impl OnnxOp for Expand {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        //https://github.com/onnx/onnx/blob/main/docs/Operators.md#Expand
        // Version 13 impl

        // unlike broadcast_to, expand allows for the output shape to
        // be different from the specified shape.
        let input_tensor = node.get_input(0)?;
        let input_shape = node.get_input(1)?;

        // Check that the shape tensor is 1D
        if input_shape.rank() != 1 {
            let err_msg = format!(
                "Expand expects 'shape' input to be 1D tensor: {:?}",
                input_shape
            );
            return Err(OnnxOpError::InvalidInput(err_msg));
        }
        let input_tensor_dims = input_tensor.dims();
        let input_shape_dims = input_shape
            .to_vec1::<i64>()?
            .into_iter()
            .map(|x| x as usize)
            .collect::<Vec<_>>();

        let target_shape = broadcast_shape(input_tensor_dims, input_shape_dims.as_slice())?;

        let expanded_tensor = input_tensor.broadcast_as(target_shape)?;

        let output_name = node.get_output(0)?;
        Ok((output_name.clone(), expanded_tensor))
    }
}
