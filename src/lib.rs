mod afn;
mod cut_weights;
mod lsh;
mod points;
mod spanning_tree;
mod ultrametric;
mod union_find;

pub use cut_weights::{CwParams, MultiplyMode};
pub use spanning_tree::KtParams;
pub use ultrametric::Ultrametric;

use numpy::{Ix2, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

/// Formats the sum of two numbers as string.
#[pyfunction]
pub fn compute_clustering<'py>(
    points: PyReadonlyArrayDyn<'py, f32>,
    c: f32,
    method: &str,
) -> PyResult<PyUltrametric> {
    PyUltrametric::new(points, c, method)
}

/// A Python module implemented in Rust.
#[pymodule]
pub fn flashcluster(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_clustering, m)?)?;
    m.add_class::<PyUltrametric>()?;
    Ok(())
}

#[pyclass(name = "Ultrametric")]
pub struct PyUltrametric {
    inner: Ultrametric,
}

#[pymethods]
impl PyUltrametric {
    #[new]
    fn new<'py>(points: PyReadonlyArrayDyn<'py, f32>, c: f32, method: &str) -> PyResult<Self> {
        let points: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, Ix2> = points
            .as_array()
            .into_dimensionality()
            .map_err(|_| PyErr::new::<PyRuntimeError, _>("Expected two-dimensional array"))?;
        Ok(Self {
            inner: Ultrametric::new(
                &points,
                KtParams { gamma: c.sqrt() },
                CwParams {
                    alpha: c.sqrt(),
                    mode: match method {
                        "precise" => MultiplyMode::Theoretical,
                        _ => MultiplyMode::SquareRoot,
                    },
                },
            ),
        })
    }

    pub fn dist(&self, i: usize, j: usize) -> f32 {
        self.inner.dist(i, j)
    }
}
