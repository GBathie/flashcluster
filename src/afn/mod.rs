//! Approximate farthest neigbhor data structure,
//! based on the work of Pagh et al.
//! (Approximate furthest neighbor with application to annulus query, 2016).

use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
    slice,
};

use itertools::Itertools;
use ndarray::{Array2, ArrayView1};
use ndarray_rand::{RandomExt, rand_distr::StandardNormal};
use ordered_float::OrderedFloat;

use crate::points::{PointId, PointSet, dist};

/// Dynamic alpha-approximate farthest neighbor data structure.
pub struct ApproxFarthestNeighbor<'pts> {
    points: &'pts PointSet,
    projections: Array2<f32>,
    m: usize,
}

impl<'pts> ApproxFarthestNeighbor<'pts> {
    pub fn new(points: &'pts PointSet, alpha: f32) -> Self {
        let (n, d) = points.dim();

        let l = (n as f32).powf(1. / alpha.powi(2)) as usize;
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

    pub fn create_clusters(&self) -> Vec<AfnCluster> {
        self.projections
            .rows()
            .into_iter()
            .enumerate()
            .map(|(id, proj)| AfnCluster::new(self.points, &self.projections, self.m, id, proj))
            .collect()
    }
}

pub struct AfnCluster<'afn> {
    points: &'afn PointSet,
    projections: &'afn Array2<f32>,
    buckets: Vec<Vec<(Reverse<OrderedFloat<f32>>, PointId)>>,
    m: usize,
}

impl<'afn> AfnCluster<'afn> {
    /// Creates a cluster containing a single point.
    pub fn new(
        points: &'afn PointSet,
        projections: &'afn Array2<f32>,
        m: usize,
        id: PointId,
        proj: ArrayView1<f32>,
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
    pub fn new_full(points: &'afn PointSet, projections: &'afn Array2<f32>, m: usize) -> Self {
        let (_, d) = projections.dim();
        let buckets = (0..d)
            .map(|i| {
                let bucket = projections
                    .rows()
                    .into_iter()
                    .enumerate()
                    .map(|(id, p)| (Reverse(OrderedFloat(p[i])), id))
                    .sorted_unstable()
                    .take(m)
                    .collect_vec();

                bucket
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

    pub fn get_farthest(&self, id: PointId) -> (PointId, f32) {
        let p = self.points.row(id);
        let projected = self.projections.row(id);
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();

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
struct HeapEntry<'a> {
    value: OrderedFloat<f32>,
    point_id: PointId,
    offset: f32,
    bucket_iter: slice::Iter<'a, (Reverse<OrderedFloat<f32>>, usize)>,
}

impl<'a> HeapEntry<'a> {
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

impl<'a> PartialEq for HeapEntry<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.point_id == other.point_id && self.offset == other.offset
    }
}

impl<'a> Eq for HeapEntry<'a> {}

impl<'a> PartialOrd for HeapEntry<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.value.cmp(&other.value))
    }
}

impl<'a> Ord for HeapEntry<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

/// 2-Approx for the diameter: find farthest point of farthest point of an arbitrary point.
///
/// The diameter is less than the returned value, and the returned value is at most twice the diameter.
pub fn estimate_diameter(points: &PointSet) -> f32 {
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
    2.0 * apx.0
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
