use candle_core::bail;

pub fn broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> candle_core::Result<Vec<usize>> {
    let (longest, shortest) = if shape_a.len() > shape_b.len() {
        (shape_a, shape_b)
    } else {
        (shape_b, shape_a)
    };
    let diff = longest.len() - shortest.len();
    let mut target_shape = longest[0..diff].to_vec();
    for (dim1, dim2) in longest[diff..].iter().zip(shortest.iter()) {
        if *dim1 == *dim2 || *dim2 == 1 || *dim1 == 1 {
            target_shape.push(usize::max(*dim1, *dim2));
        } else {
            bail!(
                "Expand: incompatible shapes for broadcast, {:?} and {:?}",
                shape_a,
                shape_b
            );
        }
    }
    Ok(target_shape)
}

pub fn broadcast_shape_from_many(shapes: &[&[usize]]) -> candle_core::Result<Vec<usize>> {
    if shapes.is_empty() {
        return Ok(Vec::new());
    }
    let mut shape_out = shapes[0].to_vec();
    for shape in shapes[1..].iter() {
        shape_out = broadcast_shape(&shape_out, shape)?;
    }
    Ok(shape_out)
}
