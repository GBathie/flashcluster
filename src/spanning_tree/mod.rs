use kt::gamma_kt;
use ndarray::Array2;
use ordered_float::OrderedFloat;

use crate::{afn::estimate_diameter, union_find::UnionFind};

mod kt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MstParams {
    pub gamma: f32,
}

/// Represents an edge as a tuple of (endpoint 1, endpoint 2, weight).
#[derive(Debug, Clone, Copy)]
pub struct Edge(pub usize, pub usize, pub f32);

#[derive(Debug, Clone)]
/// A minimum spanning tree, with edges sorted by weights.
pub struct SpanningTree {
    pub edges: Vec<Edge>,
}

impl MstParams {
    pub fn compute_mst(&self, points: &Array2<f32>) -> SpanningTree {
        let n = points.nrows();
        let max_dist = estimate_diameter(points);

        // TODO: fix min dist?
        let edges = gamma_kt(points, self.gamma, 0.01, max_dist);
        let res = exact_mst_krusal(edges, n);

        assert_eq!(res.edges.len(), n - 1);
        res
    }
}

/// Compute an exact MST using Kruskal's algorithm.
fn exact_mst_krusal(mut edges: Vec<Edge>, n: usize) -> SpanningTree {
    edges.sort_unstable_by_key(|e| OrderedFloat(e.2));
    let mut uf = UnionFind::new(n);
    let edges = edges
        .into_iter()
        .filter(|Edge(u, v, _)| uf.merge(*u, *v).is_some())
        .collect();

    SpanningTree { edges }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_mst() {
        panic!("Implement more tests!")
    }
}
