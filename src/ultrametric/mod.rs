use std::mem::swap;

use ndarray::Data;
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use ordered_float::OrderedFloat;
use rmq::Rmq;

use crate::{
    cut_weights::CwParams,
    points::{FloatType, PointSet},
    spanning_tree::{Edge, KtParams},
    union_find::UnionFindWithData,
};

mod rmq;

#[derive(Debug)]
pub struct Ultrametric<F: FloatType> {
    id_to_pos: Vec<usize>,
    rmq: Rmq<F>,
}

impl<F: FloatType> Ultrametric<F> {
    /// Compute an approximate ultrametric for the given point set.
    ///
    /// `points`: ndarray of shape (n,d) where n is the number of points, d the dimension of the space.
    pub fn new<D: Data<Elem = F>>(
        points: &PointSet<D>,
        kt_params: KtParams,
        cw_params: CwParams,
    ) -> Ultrametric<F>
    where
        StandardNormal: Distribution<F>,
    {
        let mst = kt_params.compute_kt(points);

        let cw = cw_params.compute_weights(points, mst);

        Ultrametric::single_linkage(cw)
    }

    pub(crate) fn single_linkage(mut cut_weights: Vec<Edge<F>>) -> Self {
        cut_weights.sort_unstable_by_key(|e| OrderedFloat(e.2));

        let n = cut_weights.len() + 1;
        let mut uf = UnionFindWithData::new(n);
        for Edge(u, v, w) in cut_weights {
            assert!(uf.merge(u, v, w).is_some())
        }

        let mut id_to_pos = vec![0; n];
        for (pos, id) in uf.iter_cluster(0).enumerate() {
            id_to_pos[id] = pos;
        }
        let weights = uf.iter_data(0).collect::<Vec<_>>();
        let rmq = Rmq::new(weights).unwrap();

        Self { id_to_pos, rmq }
    }

    pub fn dist(&self, i: usize, j: usize) -> F {
        if i == j {
            return F::zero();
        }

        let mut pos_i = self.id_to_pos[i];
        let mut pos_j = self.id_to_pos[j];

        if pos_i > pos_j {
            swap(&mut pos_i, &mut pos_j)
        }

        // SAFETY: i != j, therefore the range should not be empty.
        self.rmq.get_max(pos_i..pos_j).unwrap()
    }
}

pub trait UltrametricBase {
    fn dist(&self, i: usize, j: usize) -> f64;
}

impl UltrametricBase for Ultrametric<f32> {
    fn dist(&self, i: usize, j: usize) -> f64 {
        self.dist(i, j) as f64
    }
}

impl UltrametricBase for Ultrametric<f64> {
    fn dist(&self, i: usize, j: usize) -> f64 {
        self.dist(i, j)
    }
}
