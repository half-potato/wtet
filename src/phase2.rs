//! Phase 2: CPU fixup for remaining Delaunay violations.
//!
//! After GPU Phase 1, some faces may still violate the empty circumsphere
//! property. This module provides:
//! 1. Star splaying (CUDA algorithm port) for when the triangulation is complete
//! 2. Direct CPU bistellar flips as fallback for incomplete triangulations

use crate::types::*;
use std::collections::HashMap;

/// Perform star splaying on the CPU to fix Delaunay violations.
pub fn splay(points: &[[f32; 3]], result: &mut DelaunayResult) {
    if result.failed_verts.is_empty() {
        return;
    }

    // Use full CUDA star splaying algorithm
    crate::cpu::fix_with_star_splaying(points, result);
}

/// Fix remaining insphere violations via CPU bistellar flips (2-3 and 3-2).
///
/// Matches the CUDA GPU flipping kernel which uses both flip types:
/// - **2-3 flip**: 2 tets sharing a face → 3 tets sharing an edge (d,e)
/// - **3-2 flip**: 3 tets sharing an edge → 2 tets sharing a face
///
/// For each violated face, tries 3-2 first (reduces tet count), then 2-3.
pub fn cpu_flip_violations(points: &[[f32; 3]], result: &mut DelaunayResult, num_real_points: u32) {
    let pts64: Vec<[f64; 3]> = points
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();

    let num_real = num_real_points;
    let max_iters = 50;

    for iter in 0..max_iters {
        // Find all violated pairs
        let violations = find_flip_candidates(&pts64, &result.tets, &result.adjacency, num_real);
        if violations.is_empty() {
            if iter > 0 {
                eprintln!("[CPU-FLIP] Converged after {} iterations", iter);
            }
            return;
        }

        eprintln!("[CPU-FLIP] Iteration {}: {} violations", iter, violations.len());

        // Perform flips (one per violation, skip if already modified)
        let mut flipped = std::collections::HashSet::new();
        let mut new_tets = Vec::new();
        let mut dead = vec![false; result.tets.len()];
        let mut flip23_count = 0;
        let mut flip32_count = 0;

        for (ti, fi, opp_ti, opp_fi) in &violations {
            let ti = *ti;
            let fi = *fi;
            let opp_ti = *opp_ti;
            let opp_fi = *opp_fi;

            // Skip if either tet was already modified this iteration
            if flipped.contains(&ti) || flipped.contains(&opp_ti) {
                continue;
            }

            let tet_a = result.tets[ti];
            let tet_b = result.tets[opp_ti];

            // d = vertex opposite the shared face in tet_a
            let d = tet_a[fi];
            // e = vertex opposite the shared face in tet_b
            let e = tet_b[opp_fi];

            // The shared face: 3 vertices from tet_a excluding d
            let face_verts: Vec<u32> = tet_a.iter().copied().filter(|&v| v != d).collect();
            if face_verts.len() != 3 {
                continue;
            }
            let fv = [face_verts[0], face_verts[1], face_verts[2]];

            // ============================================================
            // Try 3-2 flip first (3 tets sharing edge → 2 tets sharing face)
            // CUDA checks this first because it reduces tet count.
            //
            // For each vertex `face_opp` on the shared face:
            //   - The other two vertices (ev1, ev2) form an edge
            //   - Check if tet_a's neighbor through face opposite `face_opp`
            //     contains vertex `e` — if so, 3 tets share edge (ev1, ev2)
            //   - Replace with 2 tets: [ev1, d, face_opp, e], [ev2, d, face_opp, e]
            // ============================================================
            let mut did_flip = false;
            let edge_combos: [(usize, usize, usize); 3] = [(0, 1, 2), (1, 2, 0), (2, 0, 1)];

            for &(opp_idx, ev1_idx, ev2_idx) in &edge_combos {
                let face_opp = fv[opp_idx];
                let ev1 = fv[ev1_idx];
                let ev2 = fv[ev2_idx];

                // Check both tet_a and tet_b neighbors for the third tet.
                // tet_c shares face {d, ev1, ev2} with tet_a (face opposite face_opp)
                // OR shares face {e, ev1, ev2} with tet_b (face opposite face_opp)
                let mut nei_ti_found = None;

                // Check tet_a's neighbor through face opposite face_opp
                if let Some(local_idx) = tet_a.iter().position(|&v| v == face_opp) {
                    let packed = result.adjacency[ti][local_idx];
                    if packed != INVALID {
                        let (nei_ti_u32, _) = decode_opp(packed);
                        let nei_ti = nei_ti_u32 as usize;
                        if nei_ti < result.tets.len() && !flipped.contains(&nei_ti) {
                            if result.tets[nei_ti].contains(&e) {
                                nei_ti_found = Some(nei_ti);
                            }
                        }
                    }
                }

                // Also check tet_b's neighbor through face opposite face_opp
                if nei_ti_found.is_none() {
                    if let Some(local_idx) = tet_b.iter().position(|&v| v == face_opp) {
                        let packed = result.adjacency[opp_ti][local_idx];
                        if packed != INVALID {
                            let (nei_ti_u32, _) = decode_opp(packed);
                            let nei_ti = nei_ti_u32 as usize;
                            if nei_ti < result.tets.len() && !flipped.contains(&nei_ti) {
                                if result.tets[nei_ti].contains(&d) {
                                    nei_ti_found = Some(nei_ti);
                                }
                            }
                        }
                    }
                }

                let nei_ti = match nei_ti_found {
                    Some(n) => n,
                    None => continue,
                };

                // Found 3 tets sharing edge (ev1, ev2):
                //   tet_a = {d, ev1, ev2, face_opp}
                //   tet_b = {e, ev1, ev2, face_opp}
                //   tet_c = {d, e, ev1, ev2}
                // Link of edge (ev1, ev2) is triangle (d, face_opp, e)
                // New tets: [ev1, d, face_opp, e] and [ev2, d, face_opp, e]

                let o1 = crate::predicates::orient3d(
                    pts64[ev1 as usize], pts64[d as usize],
                    pts64[face_opp as usize], pts64[e as usize],
                );
                let o2 = crate::predicates::orient3d(
                    pts64[ev2 as usize], pts64[d as usize],
                    pts64[face_opp as usize], pts64[e as usize],
                );

                // ev1 and ev2 must be on opposite sides of plane(d, face_opp, e)
                if o1 == 0.0 || o2 == 0.0 { continue; }
                if (o1 > 0.0) == (o2 > 0.0) { continue; }

                // Create tets with negative orient3d (gDel3D convention: orient3d < 0)
                let t1 = if o1 < 0.0 {
                    [ev1, d, face_opp, e]
                } else {
                    [ev1, face_opp, d, e]
                };
                let t2 = if o2 < 0.0 {
                    [ev2, d, face_opp, e]
                } else {
                    [ev2, face_opp, d, e]
                };

                dead[ti] = true;
                dead[opp_ti] = true;
                dead[nei_ti] = true;
                flipped.insert(ti);
                flipped.insert(opp_ti);
                flipped.insert(nei_ti);

                new_tets.push(t1);
                new_tets.push(t2);

                flip32_count += 1;
                did_flip = true;
                break;
            }

            if did_flip { continue; }

            // ============================================================
            // Try 2-3 flip (2 tets sharing face → 3 tets sharing edge)
            // ============================================================
            let (p, q, r) = (fv[0], fv[1], fv[2]);

            let all_orient = [
                crate::predicates::orient3d(pts64[p as usize], pts64[q as usize], pts64[d as usize], pts64[e as usize]),
                crate::predicates::orient3d(pts64[q as usize], pts64[r as usize], pts64[d as usize], pts64[e as usize]),
                crate::predicates::orient3d(pts64[r as usize], pts64[p as usize], pts64[d as usize], pts64[e as usize]),
            ];

            // All must be same sign (either all positive or all negative)
            let signs: Vec<i8> = all_orient.iter().map(|&o| {
                if o > 0.0 { 1 } else if o < 0.0 { -1 } else { 0 }
            }).collect();

            if signs.contains(&0) || (signs.contains(&1) && signs.contains(&-1)) {
                continue;
            }

            // gDel3D convention: orient3d < 0 = correct orientation
            let (t1, t2, t3) = if signs[0] < 0 {
                ([p, q, d, e], [q, r, d, e], [r, p, d, e])
            } else {
                ([q, p, d, e], [r, q, d, e], [p, r, d, e])
            };

            dead[ti] = true;
            dead[opp_ti] = true;
            flipped.insert(ti);
            flipped.insert(opp_ti);

            new_tets.push(t1);
            new_tets.push(t2);
            new_tets.push(t3);
            flip23_count += 1;
        }

        if new_tets.is_empty() {
            // Diagnostics: why can't we flip the remaining violations?
            let mut skip_already = 0;
            let mut fail_23_orient = 0;
            let mut fail_32_no_neighbor = 0;
            let mut fail_32_orient = 0;
            for &(ti, fi, opp_ti, opp_fi) in &violations {
                if flipped.contains(&ti) || flipped.contains(&opp_ti) {
                    skip_already += 1;
                    continue;
                }
                let tet_a = result.tets[ti];
                let tet_b = result.tets[opp_ti];
                let d = tet_a[fi];
                let e = tet_b[opp_fi];
                let fvd: Vec<u32> = tet_a.iter().copied().filter(|&v| v != d).collect();
                if fvd.len() != 3 { continue; }

                // Check 3-2
                let mut found_32 = false;
                for &opp_idx in &[0usize, 1, 2] {
                    let face_opp = fvd[opp_idx];
                    let local_idx = tet_a.iter().position(|&v| v == face_opp).unwrap();
                    let packed = result.adjacency[ti][local_idx];
                    if packed == INVALID { continue; }
                    let (nei_ti_u32, _) = decode_opp(packed);
                    let nei_ti = nei_ti_u32 as usize;
                    if nei_ti >= result.tets.len() { continue; }
                    let tet_c = result.tets[nei_ti];
                    if tet_c.contains(&e) {
                        found_32 = true;
                        fail_32_orient += 1;
                        break;
                    }
                }
                if !found_32 { fail_32_no_neighbor += 1; }

                // Check 2-3
                let (p, q, r) = (fvd[0], fvd[1], fvd[2]);
                let o = [
                    crate::predicates::orient3d(pts64[p as usize], pts64[q as usize], pts64[d as usize], pts64[e as usize]),
                    crate::predicates::orient3d(pts64[q as usize], pts64[r as usize], pts64[d as usize], pts64[e as usize]),
                    crate::predicates::orient3d(pts64[r as usize], pts64[p as usize], pts64[d as usize], pts64[e as usize]),
                ];
                let s: Vec<i8> = o.iter().map(|&v| if v > 0.0 { 1 } else if v < 0.0 { -1 } else { 0 }).collect();
                if s.contains(&0) || (s.contains(&1) && s.contains(&-1)) {
                    fail_23_orient += 1;
                }
            }
            eprintln!("[CPU-FLIP] No valid flips found, stopping (skip_already={}, fail_23_orient={}, fail_32_no_neighbor={}, fail_32_orient={})",
                skip_already, fail_23_orient, fail_32_no_neighbor, fail_32_orient);
            break;
        }

        eprintln!("[CPU-FLIP]   flips: 2-3={}, 3-2={}", flip23_count, flip32_count);

        // Remove dead tets, add new ones
        let mut kept_tets = Vec::new();
        for i in 0..result.tets.len() {
            if !dead[i] {
                kept_tets.push(result.tets[i]);
            }
        }
        kept_tets.extend_from_slice(&new_tets);
        result.tets = kept_tets;

        // Rebuild adjacency from scratch after flips
        rebuild_adjacency(result);
    }
}

/// Find all violated tet pairs suitable for flipping.
/// Returns (tet_idx, face_idx, opp_tet_idx, opp_face_idx) for each violation.
fn find_flip_candidates(
    pts64: &[[f64; 3]],
    tets: &[[u32; 4]],
    adjacency: &[[u32; 4]],
    num_real_points: u32,
) -> Vec<(usize, usize, usize, usize)> {
    let mut violations = Vec::new();
    for ti in 0..tets.len() {
        if tets[ti].iter().any(|&v| v >= num_real_points) {
            continue;
        }
        for f in 0..4usize {
            let packed = adjacency[ti][f];
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

            let orient = crate::predicates::orient3d(pa, pb, pc, pd);
            let insph = crate::predicates::insphere(pa, pb, pc, pd, pe);
            // Violation = e inside circumsphere: insphere * orient > 0
            let is_violation = if orient > 0.0 {
                insph > 0.0
            } else if orient < 0.0 {
                insph < 0.0
            } else {
                continue;
            };

            if is_violation {
                violations.push((ti, f, opp_ti as usize, opp_f as usize));
            }
        }
    }
    violations
}

/// Remove duplicate tets (same 4 vertices regardless of order).
/// The GPU can produce duplicates due to race conditions in concurrent splits.
pub fn deduplicate_tets(result: &mut DelaunayResult) {
    let n = result.tets.len();
    let mut seen: HashMap<[u32; 4], usize> = HashMap::with_capacity(n);
    let mut keep = vec![true; n];
    let mut dupes = 0;

    for i in 0..n {
        let t = result.tets[i];

        // Filter out dead tets (all vertices INVALID, from star splaying)
        if t.iter().any(|&v| v == INVALID) {
            keep[i] = false;
            dupes += 1;
            continue;
        }

        let mut sorted = t;
        sorted.sort();
        if let Some(&first) = seen.get(&sorted) {
            keep[i] = false;
            dupes += 1;
            if dupes <= 5 {
                eprintln!("[DEDUP] Dup #{}: tet[{}]={:?} == tet[{}]={:?}",
                    dupes, i, result.tets[i], first, result.tets[first]);
            }
        } else {
            seen.insert(sorted, i);
        }
    }

    if dupes == 0 {
        return;
    }

    eprintln!("[DEDUP] Removing {} duplicate tets from {} total", dupes, n);

    // Compact tets (adjacency will be rebuilt from scratch)
    let mut new_tets = Vec::with_capacity(n - dupes);
    for i in 0..n {
        if keep[i] {
            new_tets.push(result.tets[i]);
        }
    }
    result.tets = new_tets;
    result.adjacency.clear();
}

/// Rebuild adjacency from scratch by hashing faces.
/// This is a fallback for when incremental adjacency updates accumulate errors.
pub fn rebuild_adjacency(result: &mut DelaunayResult) {
    // First remove duplicate tets (GPU race conditions can produce them)
    deduplicate_tets(result);

    let n = result.tets.len();
    result.adjacency.clear();
    result.adjacency.resize(n, [INVALID; 4]);

    // Map from sorted face (3 vertex indices) → (tet_index, local_face)
    let mut face_map: HashMap<[u32; 3], (u32, u32)> = HashMap::with_capacity(n * 4);

    // Face opposite local index f = the 3 vertices NOT at index f
    let face_indices: [[usize; 3]; 4] = [
        [1, 2, 3], // face 0: opposite vertex 0
        [0, 2, 3], // face 1: opposite vertex 1
        [0, 1, 3], // face 2: opposite vertex 2
        [0, 1, 2], // face 3: opposite vertex 3
    ];

    for ti in 0..n {
        let tet = result.tets[ti];
        for f in 0..4u32 {
            let fi = face_indices[f as usize];
            let mut face = [tet[fi[0]], tet[fi[1]], tet[fi[2]]];
            face.sort();

            if let Some(&(other_ti, other_f)) = face_map.get(&face) {
                // Found matching face — link the two tets
                result.adjacency[ti][f as usize] = encode_opp(other_ti, other_f);
                result.adjacency[other_ti as usize][other_f as usize] =
                    encode_opp(ti as u32, f);
                face_map.remove(&face);
            } else {
                face_map.insert(face, (ti as u32, f));
            }
        }
    }
    // Remaining entries in face_map are boundary faces (INVALID, already set)
}

