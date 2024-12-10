use crate::ops::compute_node::ComputeNode;
use crate::ops::tensor_helper::broadcast_shape_from_many;
use crate::ops::{OnnxOp, OnnxOpError, OpOutput};
use candle_core::DType;

pub(crate) struct Add;
impl OnnxOp for Add {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let input0 = &node.get_input(0)?;
        let input1 = &node.get_input(1)?;
        let output = input0.broadcast_add(input1)?;

        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}

pub(crate) struct Sub;
impl OnnxOp for Sub {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let input0 = &node.get_input(0)?;
        let input1 = &node.get_input(1)?;
        let output = input0.broadcast_sub(input1)?;

        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}

pub(crate) struct Mul;
impl OnnxOp for Mul {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let input0 = &node.get_input(0)?;
        let input1 = &node.get_input(1)?;
        let output = input0.broadcast_mul(input1)?;

        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}

pub(crate) struct Div;
impl OnnxOp for Div {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let input0 = node.get_input(0)?;
        let input1 = node.get_input(1)?;
        let output = input0.broadcast_div(input1)?;

        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}

pub(crate) struct Exp;
impl OnnxOp for Exp {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let xs = node.get_input(0)?;
        let output = xs.exp()?;

        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}

pub(crate) struct Pow;
impl OnnxOp for Pow {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let input0 = node.get_input(0)?;
        let input1 = node.get_input(1)?;
        let output_name = node.get_output(0)?;

        // HACK: current implementation of broadcast_pow cannot handle negative base,
        // so we use powf where we can, which *does* correctly handle negative base.
        if let Ok(exp) = (|| input1.to_dtype(DType::F64)?.to_scalar::<f64>())() {
            let output = input0.powf(exp)?;
            Ok((output_name.clone(), output))
        } else {
            let output = input0.broadcast_pow(input1)?;
            Ok((output_name.clone(), output))
        }
    }
}

pub(crate) struct Sign;
impl OnnxOp for Sign {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let input = node.get_input(0)?;
        let output = input.sign()?;
        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}

pub(crate) struct Log;
impl OnnxOp for Log {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Log
        let input = node.get_input(0)?;
        let output = input.log()?;
        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}

pub(crate) struct Min;
impl OnnxOp for Min {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Min
        let mut output = node.get_input(0)?.clone();
        let all_inputs = node.get_all_inputs()?;
        for input in all_inputs.iter() {
            output = output.broadcast_minimum(input)?
        }
        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}

pub(crate) struct Where;
impl OnnxOp for Where {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Where
        let cond = node.get_input(0)?;
        let a = node.get_input(1)?;
        let b = node.get_input(2)?;

        // where_cond requires that all inputs are the same shape.
        // In contrast, the Where op in ONNX only requires that they are broadcastable.
        let shape = broadcast_shape_from_many(&[cond.dims(), a.dims(), b.dims()])?;
        let cond = cond.broadcast_as(shape.clone())?;
        let a = a.broadcast_as(shape.clone())?;
        let b = b.broadcast_as(shape)?;
        let output = cond.where_cond(&a, &b)?;

        let output_name = node.get_output(0)?;

        Ok((output_name.clone(), output))
    }
}

pub(crate) struct Abs;
impl OnnxOp for Abs {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Abs
        let input = node.get_input(0)?;
        let output = input.abs()?;
        let output_name = node.get_output(0)?;
        Ok((output_name.clone(), output))
    }
}

pub(crate) struct Cos;

impl OnnxOp for Cos {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Abs
        let input = node.get_input(0)?;
        let output = input.cos()?;
        let output_name = node.get_output(0)?;
        Ok((output_name.clone(), output))
    }
}
