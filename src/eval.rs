use crate::onnx::tensor_proto::DataType;
use crate::onnx::{self, GraphProto};
use crate::ops::{registry, ComputeNode, OnnxOpRegistry, OpOutput};
use candle_core::{bail, Result};
use once_cell::sync::Lazy;
use crate::parser;
use crate::parser::Value;
use std::collections::HashMap;

static REGISTRY: Lazy<OnnxOpRegistry> = Lazy::new(|| {
    registry().expect("failed to initialize registry")
});


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
    graph: &GraphProto,
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

    let registry = &*REGISTRY;
    // The nodes are topologically sorted so we can just process them in order.
    for node in graph.node.iter() {
        // TODO: Validate node.input for each operator.
        let op_type = node.op_type.as_str();
        {
            let onnx_op = registry.get(op_type)?;
            let cn = ComputeNode::new(node, values);
            let op_output = onnx_op.eval(&cn)?;
            match op_output {
                OpOutput::Single(name, value) => {
                    values.insert(name, value);
                }
                OpOutput::Multiple(outputs) => {
                    for (name, value) in outputs {
                        values.insert(name, value);
                    }
                }
                OpOutput::Branch(branch_name) => {
                    let sub_graph = parser::get_attr::<GraphProto>(node, branch_name.as_str())?;
                    let subgraph_outputs = simple_eval_(sub_graph, values)?;
                    for (i, out) in node.output.iter().enumerate() {
                        values.insert(
                            out.clone(),
                            subgraph_outputs
                                .get(&sub_graph.output[i].name)
                                .unwrap()
                                .clone(),
                        );
                    }
                }
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
