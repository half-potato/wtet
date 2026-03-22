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
            let insph = predicates::insphere(pa, pb, pc, pd, pe);
            // Violation = e inside circumsphere of (a,b,c,d)
            // Shewchuk convention: insphere > 0 means inside when orient > 0,
            //                      insphere < 0 means inside when orient < 0.
            // Equivalently: insphere * orient > 0 means inside.
            let is_violation = if orient > 0.0 {
                insph > 0.0
            } else if orient < 0.0 {
                insph < 0.0
            } else {
                continue;
            };

            if is_violation {
                violations += 1;
            }
        }
    }
    Ok(violations)
}

/// Find the set of vertices involved in insphere violations (for star splaying).
///
/// Returns a deduplicated sorted list of vertex IDs. For each violated face,
/// the opposite vertex (the one inside the circumsphere) is collected,
/// matching the CUDA gather kernel behavior.
pub fn find_violated_vertices(
    points: &[[f32; 3]],
    tets: &[[u32; 4]],
    adjacency: &[[u32; 4]],
    num_real_points: u32,
) -> Vec<u32> {
    let pts64: Vec<[f64; 3]> = points
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();

    let mut failed: std::collections::HashSet<u32> = std::collections::HashSet::new();
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
                continue;
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
            let insph = predicates::insphere(pa, pb, pc, pd, pe);
            // Violation = e inside circumsphere of (a,b,c,d)
            let is_violation = if orient > 0.0 {
                insph > 0.0
            } else if orient < 0.0 {
                insph < 0.0
            } else {
                continue;
            };

            if is_violation {
                // CUDA kerGatherFailedVerts (KerDivision.cu:697-739):
                // Collects the 3 vertices of the FAILING FACE (shared between the two
                // violating tets), NOT the intruder vertex. These shared vertices'
                // stars contain BOTH tets, so do_flipping can see and fix the violation.
                for vi in 0..4u32 {
                    if vi != f {
                        failed.insert(tet[vi as usize]);
                    }
                }
            }
        }
    }
    let mut result: Vec<u32> = failed.into_iter().collect();
    result.sort();
    result
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

    // Phase 2: CPU fixup (optional)
    let mut result = state.readback(device, queue).await;
    // Rebuild adjacency from scratch (GPU adjacency may be corrupted by flip races)
    phase2::rebuild_adjacency(&mut result);

    // DIAGNOSTIC: Check orientation BEFORE CPU flips
    {
        let num_real = normalized.len() as u32;
        let big = 100.0_f32;
        let mut diag_pts: Vec<[f64; 3]> = normalized.iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();
        diag_pts.push([-big as f64, -big as f64, -big as f64]);
        diag_pts.push([4.0 * big as f64, -big as f64, -big as f64]);
        diag_pts.push([-big as f64, 4.0 * big as f64, -big as f64]);
        diag_pts.push([-big as f64, -big as f64, 4.0 * big as f64]);
        let mut pos = 0usize;
        let mut neg = 0usize;
        let mut shown_pos = 0usize;
        let mut shown_neg = 0usize;
        for (ti, tet) in result.tets.iter().enumerate() {
            if tet.iter().any(|&v| v >= num_real) { continue; }
            let o = predicates::orient3d(
                diag_pts[tet[0] as usize],
                diag_pts[tet[1] as usize],
                diag_pts[tet[2] as usize],
                diag_pts[tet[3] as usize],
            );
            if o > 0.0 {
                pos += 1;
                if shown_pos < 3 {
                    eprintln!("[ORIENT-PRE-CPU] POSITIVE tet[{}]: verts={:?} orient3d={:.6e}", ti, tet, o);
                    for &v in tet { eprintln!("  v{}: {:?}", v, diag_pts[v as usize]); }
                    shown_pos += 1;
                }
            } else if o < 0.0 {
                neg += 1;
                if shown_neg < 3 {
                    eprintln!("[ORIENT-PRE-CPU] NEGATIVE tet[{}]: verts={:?} orient3d={:.6e}", ti, tet, o);
                    for &v in tet { eprintln!("  v{}: {:?}", v, diag_pts[v as usize]); }
                    shown_neg += 1;
                }
            }
        }
        eprintln!("[ORIENT-PRE-CPU] Before CPU flips: {} positive, {} negative orient3d", pos, neg);
    }

    if config.enable_splaying {
        // Detect insphere violations on CPU using rebuilt (correct) adjacency
        // The GPU gather shader uses GPU adjacency which may miss violations.
        let num_real = normalized.len() as u32;

        // Build extended points array including super-tet vertices
        // (same coordinates as GPU: gpu/mod.rs lines 151-155)
        let big = 100.0_f32;
        let mut extended_points = normalized.clone();
        extended_points.push([-big, -big, -big]);           // n+0
        extended_points.push([4.0 * big, -big, -big]);      // n+1
        extended_points.push([-big, 4.0 * big, -big]);      // n+2
        extended_points.push([-big, -big, 4.0 * big]);      // n+3
        extended_points.push([10.0 * big, 10.0 * big, 10.0 * big]); // n+4 (infinity)

        let violations = check_delaunay_quality(&normalized, &result.tets, &result.adjacency, num_real)
            .unwrap_or(0);
        if violations > 0 {
            eprintln!("[DELAUNAY] Phase 2: {} insphere violations, running CPU bistellar flips", violations);

            // Direct CPU bistellar flips (works with incomplete triangulations)
            phase2::cpu_flip_violations(&extended_points, &mut result, num_real);
            // Rebuild adjacency after CPU flips
            phase2::rebuild_adjacency(&mut result);
        }

        // After CPU bistellar flips, check for remaining violations.
        // If any remain, use star splaying to fix them.
        // CUDA: fixWithStarSplaying runs as FINAL step (GpuDelaunay.cu:95-97)
        // Iterate star splaying until convergence or max iterations.
        let max_splay_iters = 5;
        for splay_iter in 0..max_splay_iters {
            let remaining = check_delaunay_quality(&normalized, &result.tets, &result.adjacency, num_real)
                .unwrap_or(0);
            if remaining == 0 {
                break;
            }
            eprintln!("[DELAUNAY] {} violations remain, running star splaying (iter {})", remaining, splay_iter + 1);
            // Populate failed_verts with vertices involved in violations
            result.failed_verts = find_violated_vertices(
                &normalized, &result.tets, &result.adjacency, num_real,
            );
            eprintln!("[DELAUNAY] Found {} violated vertices for star splaying", result.failed_verts.len());
            // Use catch_unwind to prevent star splaying bugs from crashing
            let mut splay_result = result.clone();
            let splay_points = extended_points.clone();
            let splay_ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                phase2::splay(&splay_points, &mut splay_result);
                phase2::rebuild_adjacency(&mut splay_result);
                splay_result
            }));
            match splay_ok {
                Ok(fixed) => {
                    // Safety check: star splaying should not drastically reduce tet count.
                    // If starsToTetra failed to reconstruct most tets, the result is unusable.
                    let original_tet_count = result.tets.len();
                    let fixed_tet_count = fixed.tets.len();
                    if fixed_tet_count < original_tet_count / 2 {
                        eprintln!("[DELAUNAY] Star splaying destroyed triangulation ({} -> {} tets), keeping original",
                                  original_tet_count, fixed_tet_count);
                        break;
                    }
                    let post_violations = check_delaunay_quality(
                        &normalized, &fixed.tets, &fixed.adjacency, num_real,
                    ).unwrap_or(remaining);
                    if post_violations < remaining {
                        eprintln!("[DELAUNAY] Star splaying reduced violations: {} -> {}", remaining, post_violations);
                        result = fixed;
                    } else {
                        eprintln!("[DELAUNAY] Star splaying did not improve ({} -> {}), keeping original", remaining, post_violations);
                        break;
                    }
                }
                Err(_) => {
                    eprintln!("[DELAUNAY] Star splaying panicked, keeping original result");
                    break;
                }
            }
        }
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
