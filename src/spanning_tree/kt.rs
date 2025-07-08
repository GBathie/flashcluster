use std::{cmp::max, collections::VecDeque};

use ndarray::Data;

use crate::{
    lsh::{projection_lsh, rho},
    points::{PointSet, dist},
};

use super::Edge;

/// Returns a (gamma+o(1))-KT.
pub fn gamma_kt<D: Data<Elem = f32>>(
    points: &PointSet<D>,
    gamma: f32,
    min_dist: f32,
    max_dist: f32,
) -> Vec<Edge> {
    let mut edges = vec![];
    let mut radius = min_dist;
    let n = points.dim().0;
    let step = 1. + 5. / (n as f32).log2();
    while radius <= step * max_dist {
        iter_local_bfs(points, radius, gamma, &mut edges);
        radius *= step;
    }

    edges
}

fn iter_local_bfs<D: Data<Elem = f32>>(
    points: &PointSet<D>,
    radius: f32,
    gamma: f32,
    edges: &mut Vec<Edge>,
) {
    let (n, _d) = points.dim();
    let rho = rho(gamma);
    let nb_iter = max((n as f32).powf(rho) as usize, 1usize);

    for _ in 0..nb_iter {
        local_bfs(points, radius, gamma, edges);
    }
}

/// BFS in buckets of LSH
fn local_bfs<D: Data<Elem = f32>>(
    points: &PointSet<D>,
    radius: f32,
    gamma: f32,
    edges: &mut Vec<Edge>,
) {
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
