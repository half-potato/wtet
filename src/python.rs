//! Python bindings for gdel3d_wgpu using PyO3 and NumPy.

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use std::sync::OnceLock;
use crate::{delaunay_3d, required_limits};
use crate::types::{GDelConfig, InsertionRule};

// ============================================================================
// WGPU Device Management (Singleton)
// ============================================================================

static DEVICE_QUEUE: OnceLock<(wgpu::Device, wgpu::Queue)> = OnceLock::new();

/// Get or initialize the global WGPU device and queue.
fn get_device() -> PyResult<&'static (wgpu::Device, wgpu::Queue)> {
    Ok(DEVICE_QUEUE.get_or_init(|| {
        // Create instance
        let instance = wgpu::Instance::default();

        // Request adapter (GPU)
        let adapter = pollster::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
        ).expect("Failed to find GPU adapter. Make sure your system has a compatible GPU.");

        // Get required limits
        let limits = required_limits();

        // Request device
        let (device, queue) = pollster::block_on(
            adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("gdel3d_device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: Default::default(),
                experimental_features: Default::default(),
                trace: Default::default(),
            })
        ).expect("Failed to create GPU device");

        (device, queue)
    }))
}

// ============================================================================
// Python-wrapped Configuration
// ============================================================================

#[pyclass(name = "Config")]
#[derive(Clone)]
pub struct PyGDelConfig {
    inner: GDelConfig,
}

#[pymethods]
impl PyGDelConfig {
    #[new]
    #[pyo3(signature = (
        insertion_rule = "circumcenter",
        enable_flipping = true,
        enable_sorting = false,
        enable_hilbert_sorting = false,
        enable_splaying = true,
        max_insert_iterations = 100,
        max_flip_iterations = 10,
    ))]
    fn new(
        insertion_rule: &str,
        enable_flipping: bool,
        enable_sorting: bool,
        enable_hilbert_sorting: bool,
        enable_splaying: bool,
        max_insert_iterations: u32,
        max_flip_iterations: u32,
    ) -> PyResult<Self> {
        // Convert string to InsertionRule enum
        let rule = match insertion_rule.to_lowercase().as_str() {
            "circumcenter" => InsertionRule::Circumcenter,
            "centroid" => InsertionRule::Centroid,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Invalid insertion_rule '{}'. Must be 'circumcenter' or 'centroid'.", insertion_rule)
                ));
            }
        };

        Ok(PyGDelConfig {
            inner: GDelConfig {
                insertion_rule: rule,
                enable_flipping,
                enable_sorting,
                enable_hilbert_sorting,
                enable_splaying,
                max_insert_iterations,
                max_flip_iterations,
            },
        })
    }

    fn __repr__(&self) -> String {
        let rule_str = match self.inner.insertion_rule {
            InsertionRule::Circumcenter => "circumcenter",
            InsertionRule::Centroid => "centroid",
        };
        format!(
            "Config(insertion_rule='{}', flipping={}, sorting={}, hilbert={}, splaying={}, max_insert_iter={}, max_flip_iter={})",
            rule_str,
            self.inner.enable_flipping,
            self.inner.enable_sorting,
            self.inner.enable_hilbert_sorting,
            self.inner.enable_splaying,
            self.inner.max_insert_iterations,
            self.inner.max_flip_iterations,
        )
    }
}

// ============================================================================
// Python-wrapped Result
// ============================================================================

#[pyclass(name = "DelaunayResult")]
pub struct PyDelaunayResult {
    #[pyo3(get)]
    tets: Py<PyArray2<u32>>,

    #[pyo3(get)]
    adjacency: Py<PyArray2<u32>>,

    #[pyo3(get)]
    failed_verts: Py<PyArray1<u32>>,

    // Cached sizes
    num_tets_cached: usize,
    num_failed_cached: usize,
}

#[pymethods]
impl PyDelaunayResult {
    fn __repr__(&self, _py: Python) -> PyResult<String> {
        Ok(format!(
            "DelaunayResult(num_tets={}, failed_verts={})",
            self.num_tets_cached, self.num_failed_cached
        ))
    }

    #[getter]
    fn num_tets(&self, _py: Python) -> usize {
        self.num_tets_cached
    }

    #[getter]
    fn num_failed(&self, _py: Python) -> usize {
        self.num_failed_cached
    }
}

// ============================================================================
// Main Delaunay Function
// ============================================================================

#[pyfunction]
#[pyo3(signature = (points, config = None))]
fn delaunay<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f32>,
    config: Option<PyGDelConfig>,
) -> PyResult<Bound<'py, PyDelaunayResult>> {
    // Validate input shape
    let points_ref = points.as_array();
    let shape = points_ref.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Points must have shape (N, 3), got {:?}", shape)
        ));
    }

    let num_points = shape[0];
    if num_points < 4 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Need at least 4 points for 3D Delaunay, got {}", num_points)
        ));
    }

    // Convert NumPy array to Rust Vec<[f32; 3]>
    let points_array = points.as_array();
    let mut points_vec: Vec<[f32; 3]> = Vec::with_capacity(num_points);
    for i in 0..num_points {
        points_vec.push([
            points_array[[i, 0]],
            points_array[[i, 1]],
            points_array[[i, 2]],
        ]);
    }

    // Get GPU device
    let (device, queue) = get_device()?;

    // Get configuration
    let cfg = config.map(|c| c.inner).unwrap_or_default();

    // Run Delaunay (blocking on async)
    let result = pollster::block_on(delaunay_3d(device, queue, &points_vec, &cfg));

    // Convert result to NumPy arrays
    let num_tets = result.tets.len();
    let num_failed = result.failed_verts.len();

    // Convert tets Vec<[u32; 4]> to (M, 4) array
    let mut tets_flat = Vec::with_capacity(num_tets * 4);
    for tet in &result.tets {
        tets_flat.extend_from_slice(tet);
    }
    let tets_array = PyArray2::from_vec2_bound(
        py,
        &result.tets.iter().map(|t| t.to_vec()).collect::<Vec<_>>(),
    ).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to create tets array: {}", e)
        )
    })?;

    // Convert adjacency Vec<[u32; 4]> to (M, 4) array
    let adj_array = PyArray2::from_vec2_bound(
        py,
        &result.adjacency.iter().map(|a| a.to_vec()).collect::<Vec<_>>(),
    ).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to create adjacency array: {}", e)
        )
    })?;

    // Convert failed_verts Vec<u32> to (K,) array
    let failed_array = PyArray1::from_vec_bound(py, result.failed_verts);

    Bound::new(
        py,
        PyDelaunayResult {
            tets: tets_array.unbind(),
            adjacency: adj_array.unbind(),
            failed_verts: failed_array.unbind(),
            num_tets_cached: num_tets,
            num_failed_cached: num_failed,
        },
    )
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Decode packed adjacency value into (tet_idx, face_idx).
///
/// The encoding uses 5-bit shift: (tet_idx << 5) | face_idx.
#[pyfunction]
fn decode_adjacency(packed: u32) -> (u32, u32) {
    let tet_idx = packed >> 5;
    let face_idx = packed & 3;
    (tet_idx, face_idx)
}

/// Encode (tet_idx, face_idx) into packed adjacency value.
///
/// The encoding uses 5-bit shift: (tet_idx << 5) | face_idx.
#[pyfunction]
fn encode_adjacency(tet_idx: u32, face_idx: u32) -> u32 {
    (tet_idx << 5) | (face_idx & 3)
}

/// Initialize GPU device explicitly (optional, auto-called on first use).
///
/// Returns a string describing the GPU device.
#[pyfunction]
fn initialize_gpu() -> PyResult<String> {
    let (device, _queue) = get_device()?;
    Ok(format!(
        "GPU initialized: {}\nLimits: max_storage_buffers={}, max_bind_groups={}",
        device.features(),
        device.limits().max_storage_buffers_per_shader_stage,
        device.limits().max_bind_groups,
    ))
}

/// Get GPU device information.
///
/// Returns device info if GPU has been initialized, otherwise returns initialization message.
#[pyfunction]
fn gpu_info() -> PyResult<String> {
    if let Some((device, _queue)) = DEVICE_QUEUE.get() {
        Ok(format!(
            "GPU Device: {}\nFeatures: {}\nMax storage buffers: {}\nMax bind groups: {}",
            "WGPU Device",
            device.features(),
            device.limits().max_storage_buffers_per_shader_stage,
            device.limits().max_bind_groups,
        ))
    } else {
        Ok("GPU not yet initialized. Call initialize_gpu() or run delaunay() to initialize.".to_string())
    }
}

// ============================================================================
// Python Module Definition
// ============================================================================

#[pymodule]
fn gdel3d(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Main function
    m.add_function(wrap_pyfunction!(delaunay, m)?)?;

    // GPU management
    m.add_function(wrap_pyfunction!(initialize_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_info, m)?)?;

    // Helper functions
    m.add_function(wrap_pyfunction!(decode_adjacency, m)?)?;
    m.add_function(wrap_pyfunction!(encode_adjacency, m)?)?;

    // Classes
    m.add_class::<PyGDelConfig>()?;
    m.add_class::<PyDelaunayResult>()?;

    // Constants
    m.add("INVALID", u32::MAX)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
