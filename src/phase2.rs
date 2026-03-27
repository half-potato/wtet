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
    let initial_tet_count = result.tets.len();
    let max_tet_count = initial_tet_count * 3; // Hard cap on tet growth
    let mut best_tets = result.tets.clone();
    let mut best_adj = result.adjacency.clone();
    let mut best_violations = usize::MAX;
    let mut bw_stall_count = 0;
    let mut restart_count = 0;
    let max_restarts = 2;
    let mut iter = 0;
    let max_iters = 60; // Capped to avoid OOM; full_bw_reconstruction handles the rest

    loop {
        if iter >= max_iters { break; }
        // Safety: abort if tet count is exploding
        if result.tets.len() > max_tet_count {
            eprintln!("[CPU-FLIP] Tet count {} exceeds cap {}, restoring best", result.tets.len(), max_tet_count);
            result.tets = best_tets.clone();
            result.adjacency = best_adj.clone();
            break;
        }

        // Find all violated pairs
        let violations = find_flip_candidates(&pts64, &result.tets, &result.adjacency, num_real);
        if violations.is_empty() {
            if iter > 0 {
                eprintln!("[CPU-FLIP] Converged after {} iterations", iter);
            }
            return;
        }

        // Track best state
        if violations.len() < best_violations {
            best_tets = result.tets.clone();
            best_adj = result.adjacency.clone();
            best_violations = violations.len();
            bw_stall_count = 0;
        }

        eprintln!("[CPU-FLIP] Iteration {}: {} violations (best={})", iter, violations.len(), best_violations);

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

                // orient3d_sos: +1 = OrientPos (raw<0), -1 = OrientNeg (raw>0)
                let o1 = crate::predicates::orient3d_sos(
                    pts64[ev1 as usize], pts64[d as usize],
                    pts64[face_opp as usize], pts64[e as usize],
                    ev1, d, face_opp, e,
                );
                let o2 = crate::predicates::orient3d_sos(
                    pts64[ev2 as usize], pts64[d as usize],
                    pts64[face_opp as usize], pts64[e as usize],
                    ev2, d, face_opp, e,
                );

                // ev1 and ev2 must be on opposite sides of plane(d, face_opp, e)
                // SoS never returns 0, so no zero check needed
                if o1 == o2 { continue; }

                // Create tets with orient3d_sos == +1 (OrientPos = raw det<0 = gDel3D convention)
                let t1 = if o1 > 0 {
                    [ev1, d, face_opp, e]
                } else {
                    [ev1, face_opp, d, e]
                };
                let t2 = if o2 > 0 {
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

            // orient3d_sos: +1 = OrientPos (raw<0), -1 = OrientNeg (raw>0)
            let all_orient = [
                crate::predicates::orient3d_sos(pts64[p as usize], pts64[q as usize], pts64[d as usize], pts64[e as usize], p, q, d, e),
                crate::predicates::orient3d_sos(pts64[q as usize], pts64[r as usize], pts64[d as usize], pts64[e as usize], q, r, d, e),
                crate::predicates::orient3d_sos(pts64[r as usize], pts64[p as usize], pts64[d as usize], pts64[e as usize], r, p, d, e),
            ];

            // All must be same sign (SoS never returns 0)
            if !(all_orient[0] == all_orient[1] && all_orient[1] == all_orient[2]) {
                continue;
            }

            // gDel3D convention: orient3d_sos == +1 (OrientPos) = raw det<0 = correct orientation
            let (t1, t2, t3) = if all_orient[0] > 0 {
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
            let mut shown = 0;
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

                if shown < 5 {
                    eprintln!("[STUCK] tet_a[{}]={:?} (d={}), tet_b[{}]={:?} (e={}), face=[{},{},{}]",
                        ti, tet_a, d, opp_ti, tet_b, e, fvd[0], fvd[1], fvd[2]);
                    // Edge degrees
                    let edges = [(fvd[0], fvd[1]), (fvd[1], fvd[2]), (fvd[2], fvd[0])];
                    for &(ep, eq) in &edges {
                        let deg = count_edge_degree(ep, eq, ti, &result.tets, &result.adjacency);
                        eprintln!("[STUCK]   edge ({},{}): degree {}", ep, eq, deg);
                    }
                    // Also check edge (d,e) degree
                    let de_deg = count_edge_degree(d, e, ti, &result.tets, &result.adjacency);
                    eprintln!("[STUCK]   edge ({},{}) (d-e): degree {}", d, e, de_deg);
                    // Orient3d checks for 2-3
                    let (p, q, r) = (fvd[0], fvd[1], fvd[2]);
                    let o = [
                        crate::predicates::orient3d_sos(pts64[p as usize], pts64[q as usize], pts64[d as usize], pts64[e as usize], p, q, d, e),
                        crate::predicates::orient3d_sos(pts64[q as usize], pts64[r as usize], pts64[d as usize], pts64[e as usize], q, r, d, e),
                        crate::predicates::orient3d_sos(pts64[r as usize], pts64[p as usize], pts64[d as usize], pts64[e as usize], r, p, d, e),
                    ];
                    eprintln!("[STUCK]   2-3 orients: [{}, {}, {}]", o[0], o[1], o[2]);
                    shown += 1;
                }

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
                    crate::predicates::orient3d_sos(pts64[p as usize], pts64[q as usize], pts64[d as usize], pts64[e as usize], p, q, d, e),
                    crate::predicates::orient3d_sos(pts64[q as usize], pts64[r as usize], pts64[d as usize], pts64[e as usize], q, r, d, e),
                    crate::predicates::orient3d_sos(pts64[r as usize], pts64[p as usize], pts64[d as usize], pts64[e as usize], r, p, d, e),
                ];
                if !(o[0] == o[1] && o[1] == o[2]) {
                    fail_23_orient += 1;
                }
            }
            eprintln!("[CPU-FLIP] No violated-face flips found (skip_already={}, fail_23_orient={}, fail_32_no_neighbor={}, fail_32_orient={})",
                skip_already, fail_23_orient, fail_32_no_neighbor, fail_32_orient);

            // Fallback: Bowyer-Watson cavity re-triangulation.
            bw_stall_count += 1;
            if bw_stall_count > 8 {
                restart_count += 1;
                if restart_count > max_restarts {
                    eprintln!("[CPU-FLIP] BW exhausted {} restarts, stopping at best={}", max_restarts, best_violations);
                    result.tets = best_tets.clone();
                    result.adjacency = best_adj.clone();
                    break;
                }
                eprintln!("[CPU-FLIP] BW stalled, restart {}/{} from best ({} violations)", restart_count, max_restarts, best_violations);
                result.tets = best_tets.clone();
                result.adjacency = best_adj.clone();
                bw_stall_count = 0;
                iter += 1;
                continue;
            }
            let bw_count = bowyer_watson_fix(
                &pts64, result, num_real, &violations, iter,
            );
            if bw_count == 0 {
                eprintln!("[CPU-FLIP] No Bowyer-Watson fixes possible, restoring best ({} violations)", best_violations);
                result.tets = best_tets.clone();
                result.adjacency = best_adj.clone();
                break;
            }
            eprintln!("[CPU-FLIP] Performed {} Bowyer-Watson cavity fixes", bw_count);
            rebuild_adjacency(result);
            iter += 1;
            continue; // Re-enter main loop
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
        iter += 1;
    }
}

/// Full incremental Bowyer-Watson Delaunay construction.
///
/// When cpu_flip_violations can't resolve all violations (common in degenerate
/// grids), this rebuilds the triangulation from scratch using exact predicates.
/// Guaranteed correct for any input. O(n²) which is fine for small inputs.
pub fn full_bw_reconstruction(
    points: &[[f32; 3]],
    result: &mut DelaunayResult,
    num_real_points: u32,
) {
    let pts64: Vec<[f64; 3]> = points
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();
    let num_real = num_real_points as usize;
    let n = pts64.len(); // includes super-tet vertices

    let face_indices: [[usize; 3]; 4] = [
        [1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2],
    ];

    // Start with super-tetrahedron: vertices n-5..n-1 (the 4 super-tet points + infinity)
    // The super-tet vertices are at indices num_real..num_real+4 in the extended points
    let s0 = num_real as u32;     // (-big, -big, -big)
    let s1 = num_real as u32 + 1; // (4*big, -big, -big)
    let s2 = num_real as u32 + 2; // (-big, 4*big, -big)
    let s3 = num_real as u32 + 3; // (-big, -big, 4*big)

    // Orient the super-tet correctly (orient3d < 0 = gDel3D convention)
    let o = crate::predicates::orient3d_sos(
        pts64[s0 as usize], pts64[s1 as usize],
        pts64[s2 as usize], pts64[s3 as usize],
        s0, s1, s2, s3,
    );
    let init_tet = if o > 0 { [s0, s1, s2, s3] } else { [s1, s0, s2, s3] };

    let mut tets: Vec<[u32; 4]> = vec![init_tet];
    let mut adj: Vec<[u32; 4]> = vec![[INVALID; 4]];

    // Insert each real point using BW
    for pi in 0..num_real {
        let v = pi as u32;
        let p = pts64[pi];

        // Step 1: Find a tet whose circumsphere contains v (BW criterion)
        // For the first tet found, expand cavity from there.
        let mut seed_ti = None;
        for ti in 0..tets.len() {
            let t = tets[ti];
            if t.iter().any(|&x| x == INVALID) { continue; }
            // Check if v is inside circumsphere: orient4d < 0 = inside
            let o4 = crate::predicates::orient4d(
                pts64[t[0] as usize], pts64[t[1] as usize],
                pts64[t[2] as usize], pts64[t[3] as usize],
                p,
                t[0], t[1], t[2], t[3], v,
            );
            if o4 < 0 {
                seed_ti = Some(ti);
                break;
            }
        }
        let seed = match seed_ti {
            Some(s) => s,
            None => continue, // Point is outside all circumspheres (shouldn't happen)
        };

        // Step 2: BFS expand cavity (all tets whose circumsphere contains v)
        let mut cavity = std::collections::BTreeSet::new();
        let mut queue = vec![seed];
        cavity.insert(seed);

        while let Some(ti) = queue.pop() {
            for f in 0..4usize {
                let packed = adj[ti][f];
                if packed == INVALID { continue; }
                let (nei_ti_u32, _) = decode_opp(packed);
                let nei_ti = nei_ti_u32 as usize;
                if nei_ti >= tets.len() { continue; }
                if cavity.contains(&nei_ti) { continue; }
                let t = tets[nei_ti];
                if t.iter().any(|&x| x == INVALID) { continue; }

                let o4 = crate::predicates::orient4d(
                    pts64[t[0] as usize], pts64[t[1] as usize],
                    pts64[t[2] as usize], pts64[t[3] as usize],
                    p,
                    t[0], t[1], t[2], t[3], v,
                );
                if o4 < 0 {
                    cavity.insert(nei_ti);
                    queue.push(nei_ti);
                }
            }
        }

        // Step 3: Find boundary of cavity
        let mut boundary: Vec<([u32; 3], usize, usize)> = Vec::new(); // (face, neighbor_ti, neighbor_fi)
        for &ti in &cavity {
            let t = tets[ti];
            for f in 0..4usize {
                let packed = adj[ti][f];
                if packed == INVALID {
                    // External boundary
                    let fi = face_indices[f];
                    boundary.push(([t[fi[0]], t[fi[1]], t[fi[2]]], usize::MAX, 0));
                    continue;
                }
                let (nei_ti_u32, nei_fi) = decode_opp(packed);
                let nei_ti = nei_ti_u32 as usize;
                if !cavity.contains(&nei_ti) {
                    let fi = face_indices[f];
                    boundary.push(([t[fi[0]], t[fi[1]], t[fi[2]]], nei_ti, nei_fi as usize));
                }
            }
        }

        // Step 4: Create new tets connecting boundary faces to v
        let mut new_tets: Vec<[u32; 4]> = Vec::new();
        let mut new_adj: Vec<[u32; 4]> = Vec::new();
        let new_base = tets.len(); // new tets start at this index

        for (face, _nei_ti, _nei_fi) in &boundary {
            let (a, b, c) = (face[0], face[1], face[2]);
            let o = crate::predicates::orient3d_sos(
                pts64[a as usize], pts64[b as usize],
                pts64[c as usize], p,
                a, b, c, v,
            );
            new_tets.push(if o > 0 { [a, b, c, v] } else { [b, a, c, v] });
            new_adj.push([INVALID; 4]);
        }

        // Step 5: Link new tets to external neighbors and to each other
        // Link to external neighbors: each new tet's face opposite v corresponds
        // to the boundary face, which was adjacent to a non-cavity tet.
        for (bi, (face, nei_ti, nei_fi)) in boundary.iter().enumerate() {
            let new_ti = new_base + bi;
            let nt = new_tets[bi];
            // Find which local face of new tet matches the boundary face
            // The boundary face is {a,b,c} and the new tet is {a,b,c,v} or {b,a,c,v}
            // The face opposite v is the boundary face
            let v_pos = nt.iter().position(|&x| x == v).unwrap();
            // Link to external neighbor
            if *nei_ti != usize::MAX {
                new_adj[bi][v_pos] = encode_opp(*nei_ti as u32, *nei_fi as u32);
                adj[*nei_ti][*nei_fi] = encode_opp(new_ti as u32, v_pos as u32);
            }

            // Link new tets to each other: for each face of new tet that contains v,
            // find the matching new tet that shares that face
            let face_sorted = {
                let mut f = [face[0], face[1], face[2]];
                f.sort();
                f
            };
            for f in 0..4usize {
                if f == v_pos { continue; } // Already handled (external neighbor)
                if new_adj[bi][f] != INVALID { continue; } // Already linked
                let fi = face_indices[f];
                let my_face = {
                    let mut mf = [nt[fi[0]], nt[fi[1]], nt[fi[2]]];
                    mf.sort();
                    mf
                };
                // Find another new tet sharing this face
                for bj in (bi + 1)..boundary.len() {
                    let other_nt = new_tets[bj];
                    for f2 in 0..4usize {
                        let fi2 = face_indices[f2];
                        let other_face = {
                            let mut of = [other_nt[fi2[0]], other_nt[fi2[1]], other_nt[fi2[2]]];
                            of.sort();
                            of
                        };
                        if my_face == other_face {
                            let new_ti_j = new_base + bj;
                            new_adj[bi][f] = encode_opp(new_ti_j as u32, f2 as u32);
                            new_adj[bj][f2] = encode_opp(new_ti as u32, f as u32);
                        }
                    }
                }
            }
        }

        // Step 6: Remove cavity tets, add new tets
        for &ti in &cavity {
            tets[ti] = [INVALID, INVALID, INVALID, INVALID];
            adj[ti] = [INVALID; 4];
        }
        tets.extend_from_slice(&new_tets);
        adj.extend_from_slice(&new_adj);
    }

    // Compact: remove dead tets and remap adjacency
    let mut remap = vec![usize::MAX; tets.len()];
    let mut new_idx = 0;
    for i in 0..tets.len() {
        if tets[i].iter().any(|&v| v == INVALID) { continue; }
        remap[i] = new_idx;
        new_idx += 1;
    }

    let mut final_tets = Vec::with_capacity(new_idx);
    let mut final_adj = Vec::with_capacity(new_idx);
    for i in 0..tets.len() {
        if tets[i].iter().any(|&v| v == INVALID) { continue; }
        // Filter: only keep tets with ALL real vertices (< num_real)
        // or keep all and let the caller filter
        final_tets.push(tets[i]);
        let mut fa = adj[i];
        for f in 0..4 {
            if fa[f] != INVALID {
                let (old_ti, old_fi) = decode_opp(fa[f]);
                let new_ti = remap[old_ti as usize];
                if new_ti == usize::MAX {
                    fa[f] = INVALID;
                } else {
                    fa[f] = encode_opp(new_ti as u32, old_fi);
                }
            }
        }
        final_adj.push(fa);
    }

    eprintln!("[BW-FULL] Reconstructed: {} tets from {} real points", final_tets.len(), num_real);
    result.tets = final_tets;
    result.adjacency = final_adj;
}

/// Vertex re-insertion fix for stuck violations.
///
/// For each violated vertex, removes ALL tets containing it, then re-inserts
/// it via Bowyer-Watson cavity expansion. This avoids the "skip tets containing
/// vertex" problem that limits standard BW cavity fixes.
fn bowyer_watson_fix(
    pts64: &[[f64; 3]],
    result: &mut DelaunayResult,
    num_real: u32,
    violations: &[(usize, usize, usize, usize)],
    shuffle_seed: usize,
) -> usize {
    let face_indices: [[usize; 3]; 4] = [
        [1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2],
    ];

    // Collect unique violated vertices from BOTH sides of each violation
    let mut verts_to_reinsert = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for &(vi_ti, vi_fi, vi_opp_ti, vi_opp_fi) in violations {
        let e = result.tets[vi_opp_ti][vi_opp_fi]; // intruder
        let d = result.tets[vi_ti][vi_fi]; // other side
        if e < num_real && seen.insert(e) {
            verts_to_reinsert.push(e);
        }
        if d < num_real && seen.insert(d) {
            verts_to_reinsert.push(d);
        }
    }

    // Shuffle for non-deterministic exploration
    let seed = shuffle_seed;
    for i in (1..verts_to_reinsert.len()).rev() {
        let j = (seed.wrapping_mul(2654435761).wrapping_add(i * 7)) % (i + 1);
        verts_to_reinsert.swap(i, j);
    }

    let mut total_fixes = 0;

    // Process one vertex per round to limit memory growth
    for &v in &verts_to_reinsert {
        if total_fixes >= 1 { break; }

        // Step 1: Find all tets containing v
        let mut v_tets = std::collections::BTreeSet::new();
        for ti in 0..result.tets.len() {
            let tet = result.tets[ti];
            if tet.iter().any(|&x| x == INVALID) { continue; }
            if tet.contains(&v) {
                v_tets.insert(ti);
            }
        }
        if v_tets.is_empty() { continue; }
        if v_tets.len() > 30 { continue; } // Skip vertices with too many incident tets

        // Step 2: Find boundary of v_tets (faces shared with non-v tets)
        let mut boundary: Vec<([u32; 3], Option<usize>)> = Vec::new();
        for &ti in &v_tets {
            let tet = result.tets[ti];
            for f in 0..4usize {
                let packed = result.adjacency[ti][f];
                let (is_boundary, nei) = if packed == INVALID {
                    (true, None)
                } else {
                    let (nei_ti, _) = decode_opp(packed);
                    if v_tets.contains(&(nei_ti as usize)) {
                        (false, None)
                    } else {
                        (true, Some(nei_ti as usize))
                    }
                };
                if is_boundary {
                    let fi = face_indices[f];
                    boundary.push(([tet[fi[0]], tet[fi[1]], tet[fi[2]]], nei));
                }
            }
        }

        // Step 3: Expand cavity into neighbors whose circumsphere contains v
        let mut cavity = std::collections::BTreeSet::new();
        let mut queue: Vec<usize> = Vec::new();
        for &(_, nei) in &boundary {
            if let Some(nei_ti) = nei {
                if cavity.contains(&nei_ti) { continue; }
                if v_tets.contains(&nei_ti) { continue; }
                let nei_tet = result.tets[nei_ti];
                if nei_tet.iter().any(|&x| x == INVALID) { continue; }

                let o4 = crate::predicates::orient4d(
                    pts64[nei_tet[0] as usize], pts64[nei_tet[1] as usize],
                    pts64[nei_tet[2] as usize], pts64[nei_tet[3] as usize],
                    pts64[v as usize],
                    nei_tet[0], nei_tet[1], nei_tet[2], nei_tet[3], v,
                );
                if o4 < 0 {
                    cavity.insert(nei_ti);
                    queue.push(nei_ti);
                }
            }
        }

        // BFS expand cavity
        while let Some(ti) = queue.pop() {
            if cavity.len() > 20 { break; }
            for f in 0..4usize {
                let packed = result.adjacency[ti][f];
                if packed == INVALID { continue; }
                let (nei_ti_u32, _) = decode_opp(packed);
                let nei_ti = nei_ti_u32 as usize;
                if nei_ti >= result.tets.len() { continue; }
                if cavity.contains(&nei_ti) || v_tets.contains(&nei_ti) { continue; }
                let nei_tet = result.tets[nei_ti];
                if nei_tet.iter().any(|&x| x == INVALID) { continue; }

                let o4 = crate::predicates::orient4d(
                    pts64[nei_tet[0] as usize], pts64[nei_tet[1] as usize],
                    pts64[nei_tet[2] as usize], pts64[nei_tet[3] as usize],
                    pts64[v as usize],
                    nei_tet[0], nei_tet[1], nei_tet[2], nei_tet[3], v,
                );
                if o4 < 0 {
                    cavity.insert(nei_ti);
                    queue.push(nei_ti);
                }
            }
        }

        if cavity.len() > 20 { continue; }

        // Step 4: Build final boundary (v_tets + cavity)
        let mut all_removed = v_tets.clone();
        for &ti in &cavity {
            all_removed.insert(ti);
        }

        let mut final_boundary: Vec<[u32; 3]> = Vec::new();
        for &ti in &all_removed {
            let tet = result.tets[ti];
            for f in 0..4usize {
                let packed = result.adjacency[ti][f];
                let is_boundary = if packed == INVALID {
                    true
                } else {
                    let (nei_ti, _) = decode_opp(packed);
                    !all_removed.contains(&(nei_ti as usize))
                };
                if is_boundary {
                    let fi = face_indices[f];
                    final_boundary.push([tet[fi[0]], tet[fi[1]], tet[fi[2]]]);
                }
            }
        }

        // Step 5: Create new tets connecting boundary faces to v
        // Skip faces that contain v — these are from v_tets with INVALID adjacency
        // on faces that include v (external boundary of the triangulation).
        let mut new_tets: Vec<[u32; 4]> = Vec::new();
        for face in &final_boundary {
            let (a, b, c) = (face[0], face[1], face[2]);
            if a == v || b == v || c == v { continue; }
            let o = crate::predicates::orient3d_sos(
                pts64[a as usize], pts64[b as usize],
                pts64[c as usize], pts64[v as usize],
                a, b, c, v,
            );
            new_tets.push(if o > 0 { [a, b, c, v] } else { [b, a, c, v] });
        }
        if new_tets.is_empty() { continue; }

        // Verify no degenerate tets
        let degenerate = new_tets.iter().any(|nt| {
            let mut s = *nt; s.sort();
            s[0] == s[1] || s[1] == s[2] || s[2] == s[3]
        });
        if degenerate { continue; }

        // Apply: invalidate removed tets, append new ones
        for &ti in &all_removed {
            result.tets[ti] = [INVALID, INVALID, INVALID, INVALID];
        }
        for nt in &new_tets {
            result.tets.push(*nt);
        }

        total_fixes += 1;
    }

    total_fixes
}

/// Try composite flips: for each stuck violation, find a non-violated face whose 2-3 flip
/// immediately enables a flip on the original violated face.
///
/// This simulates the CUDA kernel's cascading behavior where one flip changes local topology,
/// enabling previously impossible flips.
fn try_composite_flips(
    pts64: &[[f64; 3]],
    result: &mut DelaunayResult,
    num_real: u32,
    violations: &[(usize, usize, usize, usize)],
) -> usize {
    let mut total_composite = 0;
    let mut modified = std::collections::HashSet::new();
    let mut diag_prep_valid = 0;
    let mut diag_prep_invalid = 0;
    let mut diag_no_face_match = 0;
    let mut diag_second_fail_23 = 0;
    let mut diag_second_fail_32 = 0;

    for &(vi_ti, vi_fi, vi_opp_ti, vi_opp_fi) in violations {
        if modified.contains(&vi_ti) || modified.contains(&vi_opp_ti) {
            continue;
        }

        let tet_a = result.tets[vi_ti];
        let tet_b = result.tets[vi_opp_ti];
        let d = tet_a[vi_fi]; // vertex opposite violated face in tet_a
        let e = tet_b[vi_opp_fi]; // vertex opposite violated face in tet_b

        // Shared face vertices
        let face_verts: Vec<u32> = tet_a.iter().copied().filter(|&v| v != d).collect();
        if face_verts.len() != 3 { continue; }

        // Try each non-violated face of tet_a as a prep flip site
        let mut did_composite = false;
        for prep_f in 0..4usize {
            if prep_f == vi_fi { continue; } // Skip the violated face itself

            let packed = result.adjacency[vi_ti][prep_f];
            if packed == INVALID { continue; }
            let (prep_opp_ti_u32, prep_opp_fi) = decode_opp(packed);
            let prep_opp_ti = prep_opp_ti_u32 as usize;
            if prep_opp_ti >= result.tets.len() { continue; }
            if modified.contains(&prep_opp_ti) { continue; }
            if prep_opp_ti == vi_opp_ti { continue; } // Can't use the other violated tet

            let tet_c = result.tets[prep_opp_ti];
            if tet_c.iter().any(|&v| v >= num_real) { continue; }

            // Prep face: between tet_a and tet_c
            let d_prep = tet_a[prep_f]; // tet_a vertex opposite prep face
            let e_prep = tet_c[prep_opp_fi as usize]; // tet_c vertex opposite prep face

            let prep_face: Vec<u32> = tet_a.iter().copied().filter(|&v| v != d_prep).collect();
            if prep_face.len() != 3 { continue; }
            let (pp, pq, pr) = (prep_face[0], prep_face[1], prep_face[2]);

            // Check if prep 2-3 flip is geometrically valid
            let prep_orient = [
                crate::predicates::orient3d_sos(
                    pts64[pp as usize], pts64[pq as usize],
                    pts64[d_prep as usize], pts64[e_prep as usize],
                    pp, pq, d_prep, e_prep,
                ),
                crate::predicates::orient3d_sos(
                    pts64[pq as usize], pts64[pr as usize],
                    pts64[d_prep as usize], pts64[e_prep as usize],
                    pq, pr, d_prep, e_prep,
                ),
                crate::predicates::orient3d_sos(
                    pts64[pr as usize], pts64[pp as usize],
                    pts64[d_prep as usize], pts64[e_prep as usize],
                    pr, pp, d_prep, e_prep,
                ),
            ];
            if !(prep_orient[0] == prep_orient[1] && prep_orient[1] == prep_orient[2]) {
                diag_prep_invalid += 1;
                continue;
            }
            diag_prep_valid += 1;

            // Prep 2-3 flip is valid. Compute the 3 new tets that replace tet_a and tet_c.
            let (t1, t2, t3) = if prep_orient[0] > 0 {
                ([pp, pq, d_prep, e_prep], [pq, pr, d_prep, e_prep], [pr, pp, d_prep, e_prep])
            } else {
                ([pq, pp, d_prep, e_prep], [pr, pq, d_prep, e_prep], [pp, pr, d_prep, e_prep])
            };

            // After the prep flip, tet_a is gone. Find which new tet contains the original
            // violated face {face_verts[0], face_verts[1], face_verts[2]}.
            let new_tets_arr = [t1, t2, t3];
            let mut found_new_ti = None;
            let mut found_new_fi = None;
            for (ni, nt) in new_tets_arr.iter().enumerate() {
                // Check if this new tet contains all 3 face vertices
                if face_verts.iter().all(|&fv| nt.contains(&fv)) {
                    // Find which local face is the original violated face
                    // (the face opposite the vertex that is NOT in face_verts)
                    for fi in 0..4 {
                        if !face_verts.contains(&nt[fi]) {
                            found_new_ti = Some(ni);
                            found_new_fi = Some(fi);
                            break;
                        }
                    }
                    break;
                }
            }

            let (new_ti_idx, new_fi) = match (found_new_ti, found_new_fi) {
                (Some(a), Some(b)) => (a, b),
                _ => { diag_no_face_match += 1; continue; }
            };

            let new_tet = new_tets_arr[new_ti_idx];
            let new_d = new_tet[new_fi]; // Vertex opposite the violated face in new tet

            // Check if a 2-3 flip between new_tet and tet_b is now valid
            let fv = [face_verts[0], face_verts[1], face_verts[2]];
            let orient2 = [
                crate::predicates::orient3d_sos(
                    pts64[fv[0] as usize], pts64[fv[1] as usize],
                    pts64[new_d as usize], pts64[e as usize],
                    fv[0], fv[1], new_d, e,
                ),
                crate::predicates::orient3d_sos(
                    pts64[fv[1] as usize], pts64[fv[2] as usize],
                    pts64[new_d as usize], pts64[e as usize],
                    fv[1], fv[2], new_d, e,
                ),
                crate::predicates::orient3d_sos(
                    pts64[fv[2] as usize], pts64[fv[0] as usize],
                    pts64[new_d as usize], pts64[e as usize],
                    fv[2], fv[0], new_d, e,
                ),
            ];

            if !(orient2[0] == orient2[1] && orient2[1] == orient2[2]) {
                diag_second_fail_23 += 1;
                // Fall through to try 3-2
            } else {
                // SUCCESS! Both flips are valid. Execute the composite.
                // Step 1: Replace tet_a and tet_c with t1, t2, t3 (prep 2-3)
                // Step 2: Replace new_tet (one of t1/t2/t3) and tet_b with 3 more tets (main 2-3)

                let (m1, m2, m3) = if orient2[0] > 0 {
                    ([fv[0], fv[1], new_d, e], [fv[1], fv[2], new_d, e], [fv[2], fv[0], new_d, e])
                } else {
                    ([fv[1], fv[0], new_d, e], [fv[2], fv[1], new_d, e], [fv[0], fv[2], new_d, e])
                };

                // Kill old tets
                result.tets[vi_ti] = [INVALID, INVALID, INVALID, INVALID]; // tet_a
                result.tets[prep_opp_ti] = [INVALID, INVALID, INVALID, INVALID]; // tet_c
                result.tets[vi_opp_ti] = [INVALID, INVALID, INVALID, INVALID]; // tet_b
                modified.insert(vi_ti);
                modified.insert(prep_opp_ti);
                modified.insert(vi_opp_ti);

                // Add new tets (prep flip produces 3, but one gets replaced by main flip's 3)
                for (ni, nt) in new_tets_arr.iter().enumerate() {
                    if ni == new_ti_idx {
                        // This tet gets replaced by the main flip
                        continue;
                    }
                    result.tets.push(*nt);
                }
                result.tets.push(m1);
                result.tets.push(m2);
                result.tets.push(m3);

                total_composite += 1;
                did_composite = true;
                break;
            }

            // Also check if a 3-2 flip is now possible on the violated face.
            // After the prep 2-3, the new tets may create a degree-3 edge
            // on the violated face that enables a 3-2 flip.
            // We check: for each edge of the violated face, do 3 of the new tets
            // (including tet_b) share that edge?
            let edge_pairs = [(fv[0], fv[1], fv[2]), (fv[1], fv[2], fv[0]), (fv[2], fv[0], fv[1])];
            for &(ev1, ev2, face_opp) in &edge_pairs {
                // After prep flip, tets sharing edge (ev1, ev2) include:
                // - tet_b (still exists)
                // - new_tet (contains violated face)
                // - possibly another new tet from the prep flip
                let mut sharing_tets = vec![new_tets_arr[new_ti_idx]]; // new_tet
                sharing_tets.push(tet_b); // tet_b
                for (ni, nt) in new_tets_arr.iter().enumerate() {
                    if ni == new_ti_idx { continue; }
                    if nt.contains(&ev1) && nt.contains(&ev2) {
                        sharing_tets.push(*nt);
                    }
                }

                if sharing_tets.len() != 3 { continue; }

                // For 3-2 flip: all 3 tets share edge (ev1, ev2), need to check if
                // the link is a triangle. The 3 "other" vertices (besides ev1, ev2)
                // form the link triangle.
                let mut link_verts = Vec::new();
                for t in &sharing_tets {
                    for &v in t.iter() {
                        if v != ev1 && v != ev2 && !link_verts.contains(&v) {
                            link_verts.push(v);
                        }
                    }
                }
                if link_verts.len() != 3 { continue; }

                // 3-2 flip: replace 3 tets sharing edge (ev1, ev2) with 2 tets sharing face (link)
                let (l0, l1, l2) = (link_verts[0], link_verts[1], link_verts[2]);
                let o32_1 = crate::predicates::orient3d_sos(
                    pts64[ev1 as usize], pts64[l0 as usize],
                    pts64[l1 as usize], pts64[l2 as usize],
                    ev1, l0, l1, l2,
                );
                let o32_2 = crate::predicates::orient3d_sos(
                    pts64[ev2 as usize], pts64[l0 as usize],
                    pts64[l1 as usize], pts64[l2 as usize],
                    ev2, l0, l1, l2,
                );
                if o32_1 == o32_2 { continue; } // Same side → not valid

                let nt1 = if o32_1 > 0 {
                    [ev1, l0, l1, l2]
                } else {
                    [ev1, l1, l0, l2]
                };
                let nt2 = if o32_2 > 0 {
                    [ev2, l0, l1, l2]
                } else {
                    [ev2, l1, l0, l2]
                };

                // Execute composite: prep 2-3 + main 3-2
                result.tets[vi_ti] = [INVALID, INVALID, INVALID, INVALID];
                result.tets[prep_opp_ti] = [INVALID, INVALID, INVALID, INVALID];
                result.tets[vi_opp_ti] = [INVALID, INVALID, INVALID, INVALID];
                modified.insert(vi_ti);
                modified.insert(prep_opp_ti);
                modified.insert(vi_opp_ti);

                // Add: 2 surviving prep tets + 2 main tets = 4 tets (from 3 original)
                for (ni, nt_val) in new_tets_arr.iter().enumerate() {
                    if ni == new_ti_idx { continue; } // Replaced by 3-2
                    result.tets.push(*nt_val);
                }
                // But we also need to remove the new_tet from the 3-2. Since it was never
                // actually added to result.tets, we just don't add it.
                result.tets.push(nt1);
                result.tets.push(nt2);

                total_composite += 1;
                did_composite = true;
                break;
            }

            if did_composite { break; }
        }

        // Also try non-violated faces of tet_b
        if !did_composite {
            for prep_f in 0..4usize {
                if prep_f == vi_opp_fi { continue; }

                let packed = result.adjacency[vi_opp_ti][prep_f];
                if packed == INVALID { continue; }
                let (prep_opp_ti_u32, prep_opp_fi) = decode_opp(packed);
                let prep_opp_ti = prep_opp_ti_u32 as usize;
                if prep_opp_ti >= result.tets.len() { continue; }
                if modified.contains(&prep_opp_ti) { continue; }
                if prep_opp_ti == vi_ti { continue; }

                let tet_c = result.tets[prep_opp_ti];
                if tet_c.iter().any(|&v| v >= num_real) { continue; }

                let d_prep = tet_b[prep_f];
                let e_prep = tet_c[prep_opp_fi as usize];

                let prep_face: Vec<u32> = tet_b.iter().copied().filter(|&v| v != d_prep).collect();
                if prep_face.len() != 3 { continue; }
                let (pp, pq, pr) = (prep_face[0], prep_face[1], prep_face[2]);

                let prep_orient = [
                    crate::predicates::orient3d_sos(
                        pts64[pp as usize], pts64[pq as usize],
                        pts64[d_prep as usize], pts64[e_prep as usize],
                        pp, pq, d_prep, e_prep,
                    ),
                    crate::predicates::orient3d_sos(
                        pts64[pq as usize], pts64[pr as usize],
                        pts64[d_prep as usize], pts64[e_prep as usize],
                        pq, pr, d_prep, e_prep,
                    ),
                    crate::predicates::orient3d_sos(
                        pts64[pr as usize], pts64[pp as usize],
                        pts64[d_prep as usize], pts64[e_prep as usize],
                        pr, pp, d_prep, e_prep,
                    ),
                ];
                if !(prep_orient[0] == prep_orient[1] && prep_orient[1] == prep_orient[2]) {
                    continue;
                }

                let (t1, t2, t3) = if prep_orient[0] > 0 {
                    ([pp, pq, d_prep, e_prep], [pq, pr, d_prep, e_prep], [pr, pp, d_prep, e_prep])
                } else {
                    ([pq, pp, d_prep, e_prep], [pr, pq, d_prep, e_prep], [pp, pr, d_prep, e_prep])
                };

                let new_tets_arr = [t1, t2, t3];
                let mut found_new_ti = None;
                let mut found_new_fi = None;
                for (ni, nt) in new_tets_arr.iter().enumerate() {
                    if face_verts.iter().all(|&fv| nt.contains(&fv)) {
                        for fi in 0..4 {
                            if !face_verts.contains(&nt[fi]) {
                                found_new_ti = Some(ni);
                                found_new_fi = Some(fi);
                                break;
                            }
                        }
                        break;
                    }
                }

                let (new_ti_idx, _new_fi) = match (found_new_ti, found_new_fi) {
                    (Some(a), Some(b)) => (a, b),
                    _ => continue,
                };

                let new_tet = new_tets_arr[new_ti_idx];
                let new_e = new_tet.iter().copied().find(|v| !face_verts.contains(v));
                let new_e = match new_e { Some(v) => v, None => continue };

                // Check 2-3 between tet_a and new_tet (which replaced tet_b)
                let fv = [face_verts[0], face_verts[1], face_verts[2]];
                let orient2 = [
                    crate::predicates::orient3d_sos(
                        pts64[fv[0] as usize], pts64[fv[1] as usize],
                        pts64[d as usize], pts64[new_e as usize],
                        fv[0], fv[1], d, new_e,
                    ),
                    crate::predicates::orient3d_sos(
                        pts64[fv[1] as usize], pts64[fv[2] as usize],
                        pts64[d as usize], pts64[new_e as usize],
                        fv[1], fv[2], d, new_e,
                    ),
                    crate::predicates::orient3d_sos(
                        pts64[fv[2] as usize], pts64[fv[0] as usize],
                        pts64[d as usize], pts64[new_e as usize],
                        fv[2], fv[0], d, new_e,
                    ),
                ];

                if orient2[0] == orient2[1] && orient2[1] == orient2[2] {
                    let (m1, m2, m3) = if orient2[0] > 0 {
                        ([fv[0], fv[1], d, new_e], [fv[1], fv[2], d, new_e], [fv[2], fv[0], d, new_e])
                    } else {
                        ([fv[1], fv[0], d, new_e], [fv[2], fv[1], d, new_e], [fv[0], fv[2], d, new_e])
                    };

                    result.tets[vi_ti] = [INVALID, INVALID, INVALID, INVALID];
                    result.tets[vi_opp_ti] = [INVALID, INVALID, INVALID, INVALID];
                    result.tets[prep_opp_ti] = [INVALID, INVALID, INVALID, INVALID];
                    modified.insert(vi_ti);
                    modified.insert(vi_opp_ti);
                    modified.insert(prep_opp_ti);

                    for (ni, nt) in new_tets_arr.iter().enumerate() {
                        if ni == new_ti_idx { continue; }
                        result.tets.push(*nt);
                    }
                    result.tets.push(m1);
                    result.tets.push(m2);
                    result.tets.push(m3);

                    total_composite += 1;
                    did_composite = true;
                    break;
                }
            }
        }
    }

    if total_composite == 0 {
        eprintln!("[COMPOSITE] prep_valid={}, prep_invalid={}, no_face_match={}, second_fail_23={}, second_fail_32={}",
            diag_prep_valid, diag_prep_invalid, diag_no_face_match, diag_second_fail_23, diag_second_fail_32);
    }
    total_composite
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

            // Use orient4d with full SoS: orient4d < 0 = violation (OrientNeg = inside circumsphere)
            let o4 = crate::predicates::orient4d(
                pts64[tet[0] as usize], pts64[tet[1] as usize],
                pts64[tet[2] as usize], pts64[tet[3] as usize],
                pts64[vb as usize],
                tet[0], tet[1], tet[2], tet[3], vb,
            );

            if o4 < 0 {
                violations.push((ti, f, opp_ti as usize, opp_f as usize));
            }
        }
    }
    violations
}

/// Count how many tets share a given edge (p, q), starting from start_ti.
/// Walks around the edge using face adjacency. Returns 0 if edge not found in start_ti.
fn count_edge_degree(
    p: u32, q: u32,
    start_ti: usize,
    tets: &[[u32; 4]],
    adjacency: &[[u32; 4]],
) -> usize {
    let tet = tets[start_ti];
    // Verify edge (p,q) is in start_ti
    if !tet.contains(&p) || !tet.contains(&q) {
        return 0;
    }
    // Find the two non-{p,q} vertices
    let others: Vec<u32> = tet.iter().copied().filter(|&v| v != p && v != q).collect();
    if others.len() != 2 { return 0; }

    // Walk in one direction: start from start_ti, entry_v = others[0]
    let mut count = 1; // Count start_ti
    let mut current_ti = start_ti;
    let mut entry_v = others[0]; // We "came from" the side of others[0]

    loop {
        let cur_tet = tets[current_ti];
        // Exit through face opposite entry_v
        let entry_pos = match cur_tet.iter().position(|&v| v == entry_v) {
            Some(pos) => pos,
            None => break,
        };
        let packed = adjacency[current_ti][entry_pos];
        if packed == INVALID { break; }
        let (next_ti_u32, _) = decode_opp(packed);
        let next_ti = next_ti_u32 as usize;
        if next_ti >= tets.len() { break; }
        if next_ti == start_ti {
            return count; // Closed loop
        }
        // Find exit_v (the non-{p,q} vertex that isn't entry_v) in next tet
        let next_tet = tets[next_ti];
        if !next_tet.contains(&p) || !next_tet.contains(&q) { break; }
        let exit_v = match next_tet.iter().copied().find(|&v| v != p && v != q && v != entry_v) {
            Some(v) => v,
            None => break,
        };
        count += 1;
        if count > 20 { break; } // Safety
        current_ti = next_ti;
        entry_v = exit_v;
    }

    // If we didn't close the loop, walk the other direction
    current_ti = start_ti;
    entry_v = others[1];
    loop {
        let cur_tet = tets[current_ti];
        let entry_pos = match cur_tet.iter().position(|&v| v == entry_v) {
            Some(pos) => pos,
            None => break,
        };
        let packed = adjacency[current_ti][entry_pos];
        if packed == INVALID { break; }
        let (next_ti_u32, _) = decode_opp(packed);
        let next_ti = next_ti_u32 as usize;
        if next_ti >= tets.len() { break; }
        if next_ti == start_ti { break; } // Already counted
        let next_tet = tets[next_ti];
        if !next_tet.contains(&p) || !next_tet.contains(&q) { break; }
        let exit_v = match next_tet.iter().copied().find(|&v| v != p && v != q && v != entry_v) {
            Some(v) => v,
            None => break,
        };
        count += 1;
        if count > 20 { break; }
        current_ti = next_ti;
        entry_v = exit_v;
    }

    count
}

/// Walk around edge (p,q) starting from start_ti, collecting all tet indices in order.
/// Returns None if the edge is on a boundary or traversal fails.
fn walk_edge_ring(
    p: u32, q: u32,
    start_ti: usize,
    tets: &[[u32; 4]],
    adjacency: &[[u32; 4]],
) -> Option<Vec<(usize, u32)>> {
    let tet = tets[start_ti];
    let others: Vec<u32> = tet.iter().copied().filter(|&v| v != p && v != q).collect();
    if others.len() != 2 { return None; }

    // Walk starting from others[0] direction
    let mut ring = Vec::new();
    let mut current_ti = start_ti;
    let mut entry_v = others[0];

    loop {
        let cur_tet = tets[current_ti];
        let exit_v = cur_tet.iter().copied().find(|&v| v != p && v != q && v != entry_v)?;
        ring.push((current_ti, exit_v));

        let entry_pos = cur_tet.iter().position(|&v| v == entry_v)?;
        let packed = adjacency[current_ti][entry_pos];
        if packed == INVALID { return None; }
        let (next_ti_u32, _) = decode_opp(packed);
        let next_ti = next_ti_u32 as usize;
        if next_ti >= tets.len() { return None; }
        if next_ti == start_ti {
            return Some(ring);
        }
        if ring.len() > 20 { return None; }
        current_ti = next_ti;
        entry_v = exit_v;
    }
}

/// Mark OPP_SPHERE_FAIL flags on faces that have insphere violations.
///
/// CUDA sets these flags during the GPU check_delaunay phase. After CPU flips
/// + rebuild_adjacency, the flags are lost. Star splaying's do_flipping needs
/// them to know which edges to process (Star.cpp:305-324 flipping() entry point).
pub fn mark_sphere_fail_flags(
    points: &[[f32; 3]],
    result: &mut DelaunayResult,
    num_real_points: u32,
) {
    let pts64: Vec<[f64; 3]> = points
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();

    for ti in 0..result.tets.len() {
        if result.tets[ti].iter().any(|&v| v >= num_real_points) {
            continue;
        }
        for f in 0..4usize {
            let packed = result.adjacency[ti][f];
            if packed == INVALID {
                continue;
            }
            let (opp_ti, opp_f) = decode_opp(packed);
            if opp_ti as usize >= result.tets.len() {
                continue;
            }
            if result.tets[opp_ti as usize].iter().any(|&v| v >= num_real_points) {
                continue;
            }

            let tet = result.tets[ti];
            let opp_tet = result.tets[opp_ti as usize];
            let vb = opp_tet[opp_f as usize];

            let o4 = crate::predicates::orient4d(
                pts64[tet[0] as usize], pts64[tet[1] as usize],
                pts64[tet[2] as usize], pts64[tet[3] as usize],
                pts64[vb as usize],
                tet[0], tet[1], tet[2], tet[3], vb,
            );

            if o4 < 0 {
                // Set sphere_fail flag on this face (bit 4)
                result.adjacency[ti][f] |= 1 << 4;
            }
        }
    }
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

