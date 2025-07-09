//! LSH by random projections.
use crate::points::{FloatType, PointSet};

use fxhash::FxHashMap;
use ndarray::{Array1, Array2, Data};
use ndarray_rand::{
    RandomExt,
    rand_distr::{Distribution, StandardNormal, Uniform},
};
use num_traits::{AsPrimitive, Float, NumCast};

const W_OVER_C: f32 = 2.;

// const P2: f32 = 0.684; // Probabilistically estimated
const MINUS_LOG_P2: f32 = 0.547_931_8;

/// Computes the LSH Projection of the given set of points, with parameters `radius` and `c`.
///
/// Returns a vector of buckets, i.e. lists of points with the same locality-sensitive hash.
pub fn projection_lsh<F: FloatType, D: Data<Elem = F>>(
    points: &PointSet<D>,
    radius: F,
    c: F,
) -> Vec<Vec<usize>>
where
    StandardNormal: Distribution<F>,
{
    let (n, d) = points.dim();
    let k: usize =
        (<F as NumCast>::from(n).unwrap().log2() / <F as From<f32>>::from(MINUS_LOG_P2)).as_();
    let w = <F as From<f32>>::from(W_OVER_C) * c;

    let proj = Array2::random((d, k), StandardNormal) / (radius * w);
    let shifts = Array1::random(k, Uniform::new(F::zero(), F::one()));

    // Project
    let projected = points.dot(&proj);
    // Add random shift
    let projected = projected + shifts;
    // Round down
    let projected = projected.mapv(|x| <F as AsPrimitive<usize>>::as_(Float::floor(x)));

    let mut buckets = FxHashMap::<_, Vec<usize>>::default();
    for (i, p) in projected.rows().into_iter().enumerate() {
        buckets.entry(p).or_default().push(i);
    }

    buckets.into_values().collect()
}

pub fn rho<F: FloatType>(c: F) -> F {
    <F as From<f32>>::from(0.6f32) / c
}
