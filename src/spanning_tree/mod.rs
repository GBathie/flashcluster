use kt::gamma_kt;
use ndarray::Data;
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use ordered_float::OrderedFloat;

use crate::{
    afn::estimate_diameter,
    points::{FloatType, PointSet},
    union_find::UnionFind,
};

mod kt;

/// Represents an edge as a tuple of (endpoint 1, endpoint 2, weight).
#[derive(Debug, Clone, Copy)]
pub struct Edge<F: FloatType>(pub usize, pub usize, pub F);

#[derive(Debug, Clone)]
/// A minimum spanning tree, with edges sorted by weights.
pub struct SpanningTree<F: FloatType> {
    pub edges: Vec<Edge<F>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KtParams {
    pub gamma: f32,
}

impl KtParams {
    pub fn compute_kt<F: FloatType, D: Data<Elem = F>>(
        &self,
        points: &PointSet<D>,
    ) -> SpanningTree<F>
    where
        StandardNormal: Distribution<F>,
    {
        let n = points.nrows();
        let max_dist = estimate_diameter(points);

        let edges = gamma_kt(
            points,
            <F as From<f32>>::from(self.gamma),
            <F as From<f32>>::from(0.01),
            max_dist,
        );
        let res = exact_mst_krusal(edges, n);

        assert_eq!(res.edges.len(), n - 1);
        res
    }
}

/// Compute an MST using Kruskal's algorithm.
fn exact_mst_krusal<F: FloatType>(mut edges: Vec<Edge<F>>, n: usize) -> SpanningTree<F> {
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
    use ndarray::Array2;
    use ndarray_rand::{RandomExt, rand_distr::StandardNormal};

    use crate::{
        KtParams, Ultrametric,
        points::dist,
        spanning_tree::{Edge, exact_mst_krusal},
    };

    /// WARNING: Stochastic test
    #[test]
    fn test_kt() {
        let distrib = StandardNormal;
        let points = Array2::random((200, 20), distrib);
        let (n, _dim) = points.dim();
        let gamma: f32 = 1.5;
        let params = KtParams { gamma };

        let mut full_edges = vec![];
        for i in 0..n {
            let p1 = points.row(i);
            for j in i + 1..n {
                let p2 = points.row(j);
                let d: f32 = dist(&p1, &p2);
                full_edges.push(Edge(i, j, d));
            }
        }
        let mst = exact_mst_krusal(full_edges, n);

        let kt = params.compute_kt(&points);
        // Constructing an ultrametric directly on the KT
        // gives us an efficient way to compute the max weight of an edge on every path.
        let path_max_dist_ultrametric = Ultrametric::single_linkage(kt.edges);
        let mut count_bad = 0;
        for e in mst.edges {
            if path_max_dist_ultrametric.dist(e.0, e.1) > gamma * e.2 {
                count_bad += 1;
            }
        }
        assert!(count_bad <= 10, "{count_bad}");
    }
}
