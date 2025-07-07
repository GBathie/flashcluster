use ndarray::{Array2, ArrayBase, Data, Ix1, Zip};

// Type aliases.
pub type PointId = usize;
pub type PointSet = Array2<f32>;

/// Compute the squared l2 distance between two points
pub fn dist2<D1, D2>(p1: &ArrayBase<D1, Ix1>, p2: &ArrayBase<D2, Ix1>) -> f32
where
    D1: Data<Elem = f32>,
    D2: Data<Elem = f32>,
{
    Zip::from(p1)
        .and(p2)
        .fold(0., |acc, a, b| acc + (a - b).powi(2))
}

/// Compute the l2 distance between two points
pub fn dist<D1, D2>(p1: &ArrayBase<D1, Ix1>, p2: &ArrayBase<D2, Ix1>) -> f32
where
    D1: Data<Elem = f32>,
    D2: Data<Elem = f32>,
{
    dist2(p1, p2).sqrt()
}
