//! Approximate farthest neigbhor data structure,
//! based on the work of Pagh et al.
//! (Approximate furthest neighbor with application to annulus query, 2016).

use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
    slice,
};

use itertools::Itertools;
use ndarray::{Array2, ArrayView1, Data};
use ndarray_rand::{
    RandomExt,
    rand_distr::{Distribution, StandardNormal},
};
use num_traits::{AsPrimitive, NumCast};
use ordered_float::{FloatCore, OrderedFloat};

use crate::points::{FloatType, PointId, PointSet, dist};

/// Dynamic alpha-approximate farthest neighbor data structure.
pub struct ApproxFarthestNeighbor<'pts, F: FloatType, D: Data<Elem = F>> {
    points: &'pts PointSet<D>,
    projections: Array2<F>,
    m: usize,
}

impl<'pts, F: FloatType, D: Data<Elem = F>> ApproxFarthestNeighbor<'pts, F, D>
where
    StandardNormal: Distribution<F>,
{
    pub fn new(points: &'pts PointSet<D>, alpha: F) -> Self
    where
        F: AsPrimitive<usize>,
    {
        let (n, d) = points.dim();

        let l: usize = (<F as NumCast>::from(n).unwrap())
            .powf(F::one() / FloatCore::powi(alpha, 2))
            .as_();
        let target_d = l;
        let proj = Array2::random((d, target_d), StandardNormal);
        let projections = points.dot(&proj);

        let m = 20 * (n.ilog2() + 1) as usize;

        Self {
            points,
            projections,
            m,
        }
    }

    pub fn create_clusters(&self) -> Vec<AfnCluster<F, D>> {
        self.projections
            .rows()
            .into_iter()
            .enumerate()
            .map(|(id, proj)| AfnCluster::new(self.points, &self.projections, self.m, id, proj))
            .collect()
    }
}

pub struct AfnCluster<'afn, F: FloatType, D: Data<Elem = F>> {
    points: &'afn PointSet<D>,
    projections: &'afn Array2<F>,
    buckets: Vec<Vec<(Reverse<OrderedFloat<F>>, PointId)>>,
    m: usize,
}

impl<'afn, F: FloatType, D: Data<Elem = F>> AfnCluster<'afn, F, D> {
    /// Creates a cluster containing a single point.
    pub fn new(
        points: &'afn PointSet<D>,
        projections: &'afn Array2<F>,
        m: usize,
        id: PointId,
        proj: ArrayView1<F>,
    ) -> Self {
        let buckets = proj
            .iter()
            .map(|x| vec![(Reverse((*x).into()), id)])
            .collect();

        Self {
            points,
            projections,
            buckets,
            m,
        }
    }

    /// Contains a cluster containing all points in the set.
    ///
    /// Used for testing purposes.
    pub fn new_full(points: &'afn PointSet<D>, projections: &'afn Array2<F>, m: usize) -> Self {
        let (_, d) = projections.dim();
        let buckets = (0..d)
            .map(|i| {
                projections
                    .rows()
                    .into_iter()
                    .enumerate()
                    .map(|(id, p)| (Reverse(OrderedFloat(p[i])), id))
                    .sorted_unstable()
                    .take(m)
                    .collect_vec()
            })
            .collect();

        Self {
            points,
            projections,
            buckets,
            m,
        }
    }

    /// Merges `rhs` into `self`, leaving rhs empty.
    pub fn merge(&mut self, rhs: &mut Self) {
        for (b, rb) in self.buckets.iter_mut().zip(rhs.buckets.drain(..)) {
            *b = b.drain(..).merge(rb.into_iter()).take(self.m).collect();
        }
    }

    pub fn get_farthest(&self, id: PointId) -> (PointId, F) {
        let p = self.points.row(id);
        let projected = self.projections.row(id);
        let mut heap: BinaryHeap<HeapEntry<F>> = BinaryHeap::new();

        for (i, &v) in projected.iter().enumerate() {
            let mut bucket_iter = self.buckets[i].iter();
            if let Some((value, point_id)) = bucket_iter.next() {
                let entry = HeapEntry {
                    value: value.0 - v,
                    point_id: *point_id,
                    offset: v,
                    bucket_iter,
                };

                heap.push(entry);
            }
        }

        let mut farthest = None;
        let mut it = 0;
        while let Some(entry) = heap.pop() {
            let dist = dist(&p, &self.points.row(entry.point_id));
            match farthest {
                Some((_, d)) => {
                    if dist > d {
                        farthest = Some((entry.point_id, dist));
                    }
                }
                None => farthest = Some((entry.point_id, dist)),
            }

            if let Some(entry) = entry.next() {
                heap.push(entry);
            }

            it += 1;
            if it > self.m {
                break;
            }
        }
        farthest.expect("`get_farthest` should not be called on a empty AFN data structure")
    }
}

#[derive(Debug)]
struct HeapEntry<'a, F: FloatType> {
    value: OrderedFloat<F>,
    point_id: PointId,
    offset: F,
    bucket_iter: slice::Iter<'a, (Reverse<OrderedFloat<F>>, usize)>,
}

impl<'a, F: FloatType> HeapEntry<'a, F> {
    pub fn next(mut self) -> Option<Self> {
        if let Some((v, id)) = self.bucket_iter.next() {
            Some(Self {
                value: v.0 - self.offset,
                point_id: *id,
                offset: self.offset,
                bucket_iter: self.bucket_iter,
            })
        } else {
            None
        }
    }
}

impl<'a, F: FloatType> PartialEq for HeapEntry<'a, F> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.point_id == other.point_id && self.offset == other.offset
    }
}

impl<'a, F: FloatType> Eq for HeapEntry<'a, F> {}

impl<'a, F: FloatType> PartialOrd for HeapEntry<'a, F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.value.cmp(&other.value))
    }
}

impl<'a, F: FloatType> Ord for HeapEntry<'a, F> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

/// 2-Approx for the diameter: find farthest point of farthest point of an arbitrary point.
///
/// The diameter is less than the returned value, and the returned value is at most twice the diameter.
pub fn estimate_diameter<F: FloatType, D: Data<Elem = F>>(points: &PointSet<D>) -> F {
    // Arbitrary point p0: point at index 0.
    let p0 = points.row(0);
    // Find the farthest point p1.
    let p1 = points
        .rows()
        .into_iter()
        .max_by_key(|p| OrderedFloat(dist(&p0, p)))
        .unwrap();
    // Find the max dist to p1.
    let apx = points
        .rows()
        .into_iter()
        .map(|p| OrderedFloat(dist(&p1, &p)))
        .max()
        .unwrap();
    <F as From<f32>>::from(2.0f32) * apx.0
}

#[cfg(test)]
mod tests {

    use rand::{Rng, rng};

    use super::*;

    /// WARNING: this is a stochastic test.
    #[test]
    fn random_points() {
        let distrib = StandardNormal;
        let points = Array2::random((500, 20), distrib);
        let (n, _dim) = points.dim();
        let c: f32 = 1.3;
        let it = 100;
        let ds = ApproxFarthestNeighbor::new(&points, c);
        let full_cluster = AfnCluster::new_full(ds.points, &ds.projections, ds.m);

        let mut ok = 0;
        let mut ratio = 0.;
        let mut rng = rng();
        for _ in 0..it {
            let id = rng.random_range(0..n);
            let pt = points.row(id);
            let (_, apx_d) = full_cluster.get_farthest(id);
            let max_dist = points
                .rows()
                .into_iter()
                .map(|p| dist(&pt, &p))
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            if apx_d >= max_dist / c {
                ok += 1;
            }

            ratio += max_dist / apx_d;
        }

        println!("Ratio: {ok}/{it}");
        println!("Avgr: {}", ratio / (it as f32));
        assert!(ok as f32 >= 0.7 * (it as f32))
    }
}
