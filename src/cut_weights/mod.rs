//! (1+eps)-approximation algorithm for the cut weights,
//! using Approximate Farthest Neighbor.

use std::mem::swap;

use ndarray::Data;
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use ordered_float::NotNan;

use crate::{
    afn::ApproxFarthestNeighbor,
    points::{FloatType, PointSet},
    spanning_tree::{Edge, SpanningTree},
    union_find::UnionFind,
};

#[derive(Debug, Clone, Copy)]
pub struct CwParams {
    pub alpha: f32,
    pub mode: MultiplyMode,
}

impl CwParams {
    pub fn compute_weights<F: FloatType, D: Data<Elem = F>>(
        &self,
        points: &PointSet<D>,
        mst: SpanningTree<F>,
    ) -> Vec<Edge<F>>
    where
        StandardNormal: Distribution<F>,
    {
        apx_cut_weights(points, mst, <F as From<f32>>::from(self.alpha), self.mode)
    }
}

/// Choose what factor to use when multiplying the distance
/// to the approximate farthest neighbor in the approx cut weights algorithm.
#[derive(Debug, Clone, Copy)]
pub enum MultiplyMode {
    /// Multiply the distances to the AFN by 1, i.e. do not change them.
    One,
    /// Multiply the distance to the AFN by `sqrt(alpha)`:
    /// this is sufficient in practice.
    SquareRoot,
    /// Multiply the distance to the AFN by `alpha` to ensure
    /// that we are over-approximating the cut weight w.h.p.
    Theoretical,
}

/// Compute an alpha-approximation of the cut weights in time `O(n^(1+1/alpha^2)`
/// using Approximate Farthest Neighbors queries.
pub(crate) fn apx_cut_weights<F: FloatType, D: Data<Elem = F>>(
    points: &PointSet<D>,
    mst: SpanningTree<F>,
    alpha: F,
    mode: MultiplyMode,
) -> Vec<Edge<F>>
where
    StandardNormal: Distribution<F>,
{
    let (n, _d) = points.dim();
    let mut uf = UnionFind::new(n);

    let afn = ApproxFarthestNeighbor::new(points, alpha);
    // Create an Afn data structure for each cluster
    let mut afns = afn.create_clusters();

    // Compute cut weight of each edge, from the shortest to the longest
    mst.edges
        .into_iter()
        .map(|Edge(u, v, _)| {
            let mut cu = uf.find(u);
            let mut cv = uf.find(v);
            if uf.cluster_size(cu) > uf.cluster_size(cv) {
                swap(&mut cu, &mut cv);
            }
            // `cu` is now the index of the smallest cluster.
            assert!(uf.cluster_size(cu) <= uf.cluster_size(cv));

            let (cv_afn, cu_afn) = if cu < cv {
                let (s1, s2) = afns[cu..=cv].split_at_mut(cv - cu);
                (&mut s2[0], &mut s1[0])
            } else {
                let (s1, s2) = afns[cv..=cu].split_at_mut(cu - cv);
                (&mut s1[0], &mut s2[0])
            };

            // Compute the cut weight by iterating over the smallest cluster
            // and querying the approximate farthest neighbor in the other cluster.
            let cw = uf
                .iter_cluster(cu)
                .map(|id| cv_afn.get_farthest(id).1)
                .max_by_key(|a| NotNan::new(*a).expect("NaN in cluster data"))
                .unwrap();

            // Merge smallest cluster into largest-
            assert_eq!(uf.merge(cv, cu), Some(cv));
            cv_afn.merge(cu_afn);

            match mode {
                MultiplyMode::One => Edge(u, v, cw),
                MultiplyMode::SquareRoot => Edge(u, v, alpha.sqrt() * cw),
                MultiplyMode::Theoretical => Edge(u, v, alpha * cw),
            }
        })
        .collect()
}
