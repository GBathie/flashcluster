//! LSH by random projections.
use crate::points::PointSet;

use fxhash::FxHashMap;
use ndarray::{Array1, Array2, Data};
use ndarray_rand::{
    RandomExt,
    rand_distr::{StandardNormal, Uniform},
};

const W_OVER_C: f32 = 2.;

// const P2: f32 = 0.684; // Probabilistically estimated
const MINUS_LOG_P2: f32 = 0.547_931_8;

/// Computes the LSH Projection of the given set of points, with parameters `radius` and `c`.
///
/// Returns a vector of buckets, i.e. lists of points with the same locality-sensitive hash.
pub fn projection_lsh<D: Data<Elem = f32>>(
    points: &PointSet<D>,
    radius: f32,
    c: f32,
) -> Vec<Vec<usize>> {
    let (n, d) = points.dim();
    let k = ((n as f32).log2() / MINUS_LOG_P2) as usize;
    let w = W_OVER_C * c;

    let proj = Array2::random((d, k), StandardNormal) / (radius * w);
    let shifts = Array1::random(k, Uniform::new(0., 1.));

    // Project
    let projected = points.dot(&proj);
    // Add random shift
    let projected = projected + shifts;
    // Round down
    let projected = projected.mapv(|x| x.floor() as u32);

    let mut buckets = FxHashMap::<_, Vec<usize>>::default();
    for (i, p) in projected.rows().into_iter().enumerate() {
        buckets.entry(p).or_default().push(i);
    }

    buckets.into_values().collect()
}

pub fn rho(c: f32) -> f32 {
    0.6 / c
}
