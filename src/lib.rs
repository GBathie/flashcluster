mod afn;
mod cut_weights;
mod lsh;
mod points;
mod spanning_tree;
mod ultrametric;
mod union_find;

use std::sync::Arc;

pub use cut_weights::{CwParams, MultiplyMode};
pub use spanning_tree::KtParams;
pub use ultrametric::Ultrametric;

use numpy::PyReadonlyArray2;
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::ultrametric::UltrametricBase;

#[derive(FromPyObject)]
enum PyArrayf32orf64<'py> {
    F32(PyReadonlyArray2<'py, f32>),
    F64(PyReadonlyArray2<'py, f64>),
}

#[pyfunction]
pub fn compute_clustering<'py>(
    points: &Bound<'py, PyAny>,
    c: f32,
    mode: &str,
) -> PyResult<PyUltrametric> {
    let points = points.extract().map_err(|_| {
        PyErr::new::<PyValueError, _>(
            "Expected two-dimensional array with np.float32 or np.float64 values:",
        )
    })?;
    PyUltrametric::new(points, c, mode)
}

#[pymodule]
pub fn flashcluster(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_clustering, m)?)?;
    m.add_class::<PyUltrametric>()?;
    Ok(())
}

#[pyclass(name = "Ultrametric")]
pub struct PyUltrametric {
    inner: Arc<dyn UltrametricBase + Sync + Send>,
}

#[pymethods]
impl PyUltrametric {
    #[new]
    fn new<'py>(points: PyArrayf32orf64<'py>, c: f32, mode: &str) -> PyResult<Self> {
        if c < 1.0 {
            return Err(PyErr::new::<PyValueError, _>(
                "Value of c must be at least 1.0",
            ));
        }
        let kt_params = KtParams { gamma: c.sqrt() };
        let cw_params = CwParams {
            alpha: c.sqrt(),
            mode: match mode {
                "precise" => MultiplyMode::Theoretical,
                _ => MultiplyMode::SquareRoot,
            },
        };

        let um = match points {
            PyArrayf32orf64::F32(points) => Self {
                inner: Arc::from(Ultrametric::new(&points.as_array(), kt_params, cw_params)),
            },
            PyArrayf32orf64::F64(points) => Self {
                inner: Arc::from(Ultrametric::new(&points.as_array(), kt_params, cw_params)),
            },
        };

        Ok(um)
    }

    pub fn dist(&self, i: usize, j: usize) -> f64 {
        self.inner.dist(i, j)
    }
}
