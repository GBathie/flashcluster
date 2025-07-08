use std::mem::swap;

use ndarray::Data;
use ordered_float::OrderedFloat;
use rmq::Rmq;

use crate::{
    cut_weights::CwParams,
    points::PointSet,
    spanning_tree::{Edge, KtParams},
    union_find::UnionFindWithData,
};

mod rmq;

#[derive(Debug)]
pub struct Ultrametric {
    id_to_pos: Vec<usize>,
    rmq: Rmq,
}

impl Ultrametric {
    /// Compute an approximate ultrametric for the given point set.
    ///
    /// `points`: ndarray of shape (n,d) where n is the number of points, d the dimension of the space.
    pub fn new<D: Data<Elem = f32>>(
        points: &PointSet<D>,
        kt_params: KtParams,
        cw_params: CwParams,
    ) -> Ultrametric {
        let mst = kt_params.compute_kt(points);

        let cw = cw_params.compute_weights(points, mst);

        Ultrametric::single_linkage(cw)
    }

    pub(crate) fn single_linkage(mut cut_weights: Vec<Edge>) -> Self {
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

    pub fn dist(&self, i: usize, j: usize) -> f32 {
        if i == j {
            return 0.;
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
