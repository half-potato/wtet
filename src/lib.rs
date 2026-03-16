//! GPU-accelerated 3D Delaunay triangulation using wgpu compute shaders.
//!
//! Port of gdel3D (Cao Thanh Tung, NUS 2014) to Rust + WGSL.

pub mod types;
pub mod predicates;
pub mod gpu;
pub mod phase1;
pub mod phase2;
pub mod cpu;
pub mod profiler;
pub mod morton;
pub mod hilbert;

#[cfg(feature = "python")]
pub mod python;

use types::{DelaunayResult, GDelConfig, INVALID, decode_opp};

/// Check Delaunay property for interior tet pairs only (skip super-tet vertices).
///
/// Super-tet vertices use large finite coordinates that can cause false
/// insphere results, so we only check pairs where both tets are fully interior.
/// Returns `Ok(num_violations)` or `Err` on structural problems.
pub fn check_delaunay_quality(
    points: &[[f32; 3]],
    tets: &[[u32; 4]],
    adjacency: &[[u32; 4]],
    num_real_points: u32,
) -> Result<usize, String> {
    let pts64: Vec<[f64; 3]> = points
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();

    let mut violations = 0;
    for ti in 0..tets.len() {
        if tets[ti].iter().any(|&v| v >= num_real_points) {
            continue;
        }
        for f in 0..4u32 {
            let packed = adjacency[ti][f as usize];
            if packed == INVALID {
                continue;
            }
            let (opp_ti, opp_f) = decode_opp(packed);
            if ti >= opp_ti as usize {
                continue;
            }
            if opp_ti as usize >= tets.len() {
                return Err(format!(
                    "Tet {ti} face {f}: neighbor {opp_ti} out of range (len={})",
                    tets.len()
                ));
            }
            if tets[opp_ti as usize].iter().any(|&v| v >= num_real_points) {
                continue;
            }

            let tet = tets[ti];
            let opp_tet = tets[opp_ti as usize];
            let vb = opp_tet[opp_f as usize];

            let pa = pts64[tet[0] as usize];
            let pb = pts64[tet[1] as usize];
            let pc = pts64[tet[2] as usize];
            let pd = pts64[tet[3] as usize];
            let pe = pts64[vb as usize];

            let orient = predicates::orient3d(pa, pb, pc, pd);
            let insph = if orient > 0.0 {
                predicates::insphere(pa, pb, pc, pd, pe)
            } else if orient < 0.0 {
                -predicates::insphere(pa, pc, pb, pd, pe)
            } else {
                continue;
            };

            if insph > 0.0 {
                violations += 1;
            }
        }
    }
    Ok(violations)
}

/// Returns the minimum wgpu limits required by gdel3d_wgpu.
pub fn required_limits() -> wgpu::Limits {
    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffers_per_shader_stage = 20;  // Increased for block_owner + debug buffers
    limits.max_bind_groups = 4;
    limits
}

/// Compute the 3D Delaunay triangulation of `points` on the GPU.
///
/// The caller provides a wgpu device and queue. The device must have been
/// created with at least `required_limits()`. Points are uploaded,
/// the GPU kernels run, and the result is read back to the CPU.
pub async fn delaunay_3d(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    points: &[[f32; 3]],
    config: &GDelConfig,
) -> DelaunayResult {
    eprintln!("\n[DELAUNAY] Starting delaunay_3d with {} points", points.len());
    assert!(points.len() >= 4, "Need at least 4 points for 3D Delaunay");

    // Normalize points to [0,1]³ for numerical stability
    eprintln!("[DELAUNAY] Normalizing points...");
    let (normalized, _bbox_min, _bbox_scale) = normalize_points(points);
    eprintln!("[DELAUNAY] ✓ Points normalized");

    // Phase 1: GPU insertion + flipping
    eprintln!("[DELAUNAY] Creating GpuState...");
    let mut state = gpu::GpuState::new(device, queue, &normalized, config).await;
    eprintln!("[DELAUNAY] ✓ GpuState created");
    phase1::run(device, queue, &mut state, config).await;

    // Phase 2: CPU star splaying (optional)
    let mut result = state.readback(device, queue).await;
    if config.enable_splaying && !result.failed_verts.is_empty() {
        phase2::splay(&normalized, &mut result);
    }

    result
}

/// Normalize points to [0,1]³, returns (normalized_points, bbox_min, bbox_scale).
fn normalize_points(points: &[[f32; 3]]) -> (Vec<[f32; 3]>, [f32; 3], f32) {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for p in points {
        for i in 0..3 {
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }
    }
    let extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    let scale = extent[0].max(extent[1]).max(extent[2]).max(1e-10);

    let normalized: Vec<[f32; 3]> = points
        .iter()
        .map(|p| {
            [
                (p[0] - min[0]) / scale,
                (p[1] - min[1]) / scale,
                (p[2] - min[2]) / scale,
            ]
        })
        .collect();

    (normalized, min, scale)
}

#[cfg(test)]
mod tests;
