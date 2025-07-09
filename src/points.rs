use ndarray::{ArrayBase, Data, Ix1, Ix2, ScalarOperand, Zip};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num_traits::{AsPrimitive, Float};
use ordered_float::FloatCore;

// Type aliases.
pub type PointId = usize;
pub type PointSet<D /*: Data<Elem = F: FloatType>*/> = ArrayBase<D, Ix2>;

pub trait FloatType:
    'static
    + Float
    + FloatCore
    + AsPrimitive<usize>
    + AsPrimitive<f64>
    + From<f32>
    + ScalarOperand
    + SampleUniform
{
}

impl<T> FloatType for T where
    T: 'static
        + Float
        + FloatCore
        + AsPrimitive<usize>
        + AsPrimitive<f64>
        + From<f32>
        + ScalarOperand
        + SampleUniform
{
}

/// Compute the squared l2 distance between two points
pub fn dist2<F: FloatType, D1, D2>(p1: &ArrayBase<D1, Ix1>, p2: &ArrayBase<D2, Ix1>) -> F
where
    D1: Data<Elem = F>,
    D2: Data<Elem = F>,
{
    Zip::from(p1)
        .and(p2)
        .fold(F::zero(), |acc, a, b| acc + Float::powi(*a + b.neg(), 2))
}

/// Compute the l2 distance between two points
pub fn dist<F: FloatType, D1, D2>(p1: &ArrayBase<D1, Ix1>, p2: &ArrayBase<D2, Ix1>) -> F
where
    D1: Data<Elem = F>,
    D2: Data<Elem = F>,
{
    dist2(p1, p2).sqrt()
}
