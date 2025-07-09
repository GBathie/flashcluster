use std::{cmp::max, collections::VecDeque};

use ndarray::Data;
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use num_traits::NumCast;

use crate::{
    lsh::{projection_lsh, rho},
    points::{FloatType, PointSet, dist},
};

use super::Edge;

/// Returns a (gamma+o(1))-KT.
pub fn gamma_kt<F: FloatType, D: Data<Elem = F>>(
    points: &PointSet<D>,
    gamma: F,
    min_dist: F,
    max_dist: F,
) -> Vec<Edge<F>>
where
    StandardNormal: Distribution<F>,
{
    let mut edges = vec![];
    let mut radius = min_dist;
    let n = points.dim().0;
    let step = F::one() + <F as From<f32>>::from(5.0) / (<F as NumCast>::from(n).unwrap()).log2();
    while radius <= step * max_dist {
        iter_local_bfs(points, radius, gamma, &mut edges);
        radius = radius * step;
    }

    edges
}

fn iter_local_bfs<F: FloatType, D: Data<Elem = F>>(
    points: &PointSet<D>,
    radius: F,
    gamma: F,
    edges: &mut Vec<Edge<F>>,
) where
    StandardNormal: Distribution<F>,
{
    let (n, _d) = points.dim();
    let rho = rho(gamma);
    let nb_iter = max(<F as NumCast>::from(n).unwrap().powf(rho).as_(), 1usize);

    for _ in 0..nb_iter {
        local_bfs(points, radius, gamma, edges);
    }
}

/// BFS in buckets of LSH
fn local_bfs<F: FloatType, D: Data<Elem = F>>(
    points: &PointSet<D>,
    radius: F,
    gamma: F,
    edges: &mut Vec<Edge<F>>,
) where
    StandardNormal: Distribution<F>,
{
    let buckets = projection_lsh(points, radius, gamma);
    for mut b in buckets {
        while let Some(x) = b.pop() {
            let mut q = VecDeque::new();
            q.push_back(x);
            while let Some(u) = q.pop_front() {
                let p_u = points.row(u);
                // Iterate over b, retain only elements that are far,
                // and use a side effect to add edge to the others.
                b.retain(|&v| {
                    let p_v = points.row(v);
                    let d = dist(&p_u, &p_v);
                    if d <= gamma * radius {
                        edges.push(Edge(u, v, d));
                        false
                    } else {
                        true
                    }
                });
            }
        }
    }
}
