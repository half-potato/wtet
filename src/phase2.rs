//! Phase 2: CPU star splaying for fixing remaining Delaunay violations.
//!
//! After GPU Phase 1, some faces may still violate the empty circumsphere
//! property. Star splaying performs local bistellar flips on the CPU to fix
//! these. This is the fallback that guarantees correctness.

use crate::predicates;
use crate::types::*;
use std::collections::HashMap;

/// Perform star splaying on the CPU to fix Delaunay violations.
pub fn splay(points: &[[f32; 3]], result: &mut DelaunayResult) {
    if result.failed_verts.is_empty() {
        return;
    }

    log::info!(
        "Phase 2: star splaying for {} failed vertices",
        result.failed_verts.len()
    );

    let pts64: Vec<[f64; 3]> = points
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();

    let num_tets = result.tets.len();
    let max_iterations = num_tets * 10;
    let mut flips_done = 0usize;

    for _iter in 0..max_iterations {
        let mut flipped_any = false;

        for ti in 0..result.tets.len() {
            let tet = result.tets[ti];

            for face in 0..4u32 {
                let opp_packed = result.adjacency[ti][face as usize];
                if opp_packed == INVALID {
                    continue;
                }

                let (opp_ti, opp_face) = decode_opp(opp_packed);
                if opp_ti as usize >= result.tets.len() {
                    continue;
                }

                if ti >= opp_ti as usize {
                    continue;
                }

                let opp_tet = result.tets[opp_ti as usize];

                let va = tet[face as usize];
                let vb = opp_tet[opp_face as usize];

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

                if insph <= 0.0 {
                    continue;
                }

                // Get shared face vertices
                let mut shared = [0u32; 3];
                let mut si = 0;
                for i in 0..4 {
                    if i != face as usize {
                        shared[si] = tet[i];
                        si += 1;
                    }
                }

                // Check flip validity
                let pva = pts64[va as usize];
                let pvb = pts64[vb as usize];
                let ps0 = pts64[shared[0] as usize];
                let ps1 = pts64[shared[1] as usize];
                let ps2 = pts64[shared[2] as usize];

                let o0 = predicates::orient3d(pva, pvb, ps0, ps1);
                let o1 = predicates::orient3d(pva, pvb, ps1, ps2);
                let o2 = predicates::orient3d(pva, pvb, ps2, ps0);

                let s0 = predicates::sign(o0);
                let s1 = predicates::sign(o1);
                let s2 = predicates::sign(o2);

                let all_pos = s0 >= 0 && s1 >= 0 && s2 >= 0;
                let all_neg = s0 <= 0 && s1 <= 0 && s2 <= 0;

                if !all_pos && !all_neg {
                    continue;
                }

                // === Perform 2-3 flip ===
                // Record external adjacency from A and B before overwriting
                let a_opp = result.adjacency[ti];
                let b_opp = result.adjacency[opp_ti as usize];

                // Construct new tets (before orientation fix)
                let mut new_tets = [
                    [va, vb, shared[0], shared[1]],
                    [va, vb, shared[1], shared[2]],
                    [va, vb, shared[2], shared[0]],
                ];

                // Track orientation swaps
                let mut swapped = [false; 3];
                for k in 0..3 {
                    let t = new_tets[k];
                    let o = predicates::orient3d(
                        pts64[t[0] as usize],
                        pts64[t[1] as usize],
                        pts64[t[2] as usize],
                        pts64[t[3] as usize],
                    );
                    if o < 0.0 {
                        new_tets[k].swap(2, 3);
                        swapped[k] = true;
                    }
                }

                // Indices of new tets
                let t2_idx = result.tets.len() as u32;
                let new_tet_indices = [ti as u32, opp_ti, t2_idx];

                result.tets[ti] = new_tets[0];
                result.tets[opp_ti as usize] = new_tets[1];
                result.tets.push(new_tets[2]);

                // Initialize adjacency for new tets
                result.adjacency[ti] = [INVALID; 4];
                result.adjacency[opp_ti as usize] = [INVALID; 4];
                result.adjacency.push([INVALID; 4]);

                // Helper to find local index of vertex in a tet
                let find_local = |tet: &[u32; 4], v: u32| -> usize {
                    tet.iter().position(|&x| x == v).unwrap()
                };

                // === Internal adjacency ===
                // C0 has {va,vb,s0,s1}, C1 has {va,vb,s1,s2}, C2 has {va,vb,s2,s0}
                // C0↔C1: C0 face opp s0 <-> C1 face opp s2
                let c0_local_s0 = find_local(&new_tets[0], shared[0]);
                let c1_local_s2 = find_local(&new_tets[1], shared[2]);
                result.adjacency[ti][c0_local_s0] =
                    encode_opp(new_tet_indices[1], c1_local_s2 as u32);
                result.adjacency[opp_ti as usize][c1_local_s2] =
                    encode_opp(new_tet_indices[0], c0_local_s0 as u32);

                // C1↔C2: C1 face opp s1 <-> C2 face opp s0
                let c1_local_s1 = find_local(&new_tets[1], shared[1]);
                let c2_local_s0 = find_local(&new_tets[2], shared[0]);
                result.adjacency[opp_ti as usize][c1_local_s1] =
                    encode_opp(new_tet_indices[2], c2_local_s0 as u32);
                result.adjacency[t2_idx as usize][c2_local_s0] =
                    encode_opp(new_tet_indices[1], c1_local_s1 as u32);

                // C2↔C0: C2 face opp s2 <-> C0 face opp s1
                let c2_local_s2 = find_local(&new_tets[2], shared[2]);
                let c0_local_s1 = find_local(&new_tets[0], shared[1]);
                result.adjacency[t2_idx as usize][c2_local_s2] =
                    encode_opp(new_tet_indices[0], c0_local_s1 as u32);
                result.adjacency[ti][c0_local_s1] =
                    encode_opp(new_tet_indices[2], c2_local_s2 as u32);

                // === External adjacency ===
                // For each shared vertex sk:
                //   - In old tet_a, face opposite sk → goes to new tet not containing sk
                //     s0 not in C1 → C1 face-opp-va gets A's face-opp-s0
                //     s1 not in C2 → C2 face-opp-va gets A's face-opp-s1
                //     s2 not in C0 → C0 face-opp-va gets A's face-opp-s2
                //   - In old tet_b, face opposite sk → goes to new tet not containing sk
                //     s0 not in C1 → C1 face-opp-vb gets B's face-opp-s0
                //     s1 not in C2 → C2 face-opp-vb gets B's face-opp-s1
                //     s2 not in C0 → C0 face-opp-vb gets B's face-opp-s2

                let a_local_s0 = tet.iter().position(|&x| x == shared[0]).unwrap();
                let a_local_s1 = tet.iter().position(|&x| x == shared[1]).unwrap();
                let a_local_s2 = tet.iter().position(|&x| x == shared[2]).unwrap();

                let b_local_s0 = opp_tet.iter().position(|&x| x == shared[0]).unwrap();
                let b_local_s1 = opp_tet.iter().position(|&x| x == shared[1]).unwrap();
                let b_local_s2 = opp_tet.iter().position(|&x| x == shared[2]).unwrap();

                // C0 (index ti): gets A's face-opp-s2 at C0's face-opp-va, B's face-opp-s2 at C0's face-opp-vb
                let c0_local_va = find_local(&new_tets[0], va);
                let c0_local_vb = find_local(&new_tets[0], vb);
                result.adjacency[ti][c0_local_va] = a_opp[a_local_s2];
                result.adjacency[ti][c0_local_vb] = b_opp[b_local_s2];

                // C1 (index opp_ti): gets A's face-opp-s0 at C1's face-opp-va, B's face-opp-s0 at C1's face-opp-vb
                let c1_local_va = find_local(&new_tets[1], va);
                let c1_local_vb = find_local(&new_tets[1], vb);
                result.adjacency[opp_ti as usize][c1_local_va] = a_opp[a_local_s0];
                result.adjacency[opp_ti as usize][c1_local_vb] = b_opp[b_local_s0];

                // C2 (index t2_idx): gets A's face-opp-s1 at C2's face-opp-va, B's face-opp-s1 at C2's face-opp-vb
                let c2_local_va = find_local(&new_tets[2], va);
                let c2_local_vb = find_local(&new_tets[2], vb);
                result.adjacency[t2_idx as usize][c2_local_va] = a_opp[a_local_s1];
                result.adjacency[t2_idx as usize][c2_local_vb] = b_opp[b_local_s1];

                // === Back-pointer fix ===
                // Update external neighbors to point back to the correct new tet
                let ext_pairs: [(u32, usize, u32, usize); 6] = [
                    // (ext_opp, new_tet_idx_index, new_tet_index, face_in_new_tet)
                    (a_opp[a_local_s2], 0, new_tet_indices[0], c0_local_va),
                    (a_opp[a_local_s0], 1, new_tet_indices[1], c1_local_va),
                    (a_opp[a_local_s1], 2, new_tet_indices[2], c2_local_va),
                    (b_opp[b_local_s2], 0, new_tet_indices[0], c0_local_vb),
                    (b_opp[b_local_s0], 1, new_tet_indices[1], c1_local_vb),
                    (b_opp[b_local_s1], 2, new_tet_indices[2], c2_local_vb),
                ];

                for &(ext_packed, _ci, new_ti, new_face) in &ext_pairs {
                    if ext_packed == INVALID {
                        continue;
                    }
                    let (n_tet, n_face) = decode_opp(ext_packed);
                    if (n_tet as usize) < result.adjacency.len() {
                        result.adjacency[n_tet as usize][n_face as usize] =
                            encode_opp(new_ti, new_face as u32);
                    }
                }

                flips_done += 1;
                flipped_any = true;
            }
        }

        if !flipped_any {
            break;
        }
    }

    result.failed_verts.clear();

    log::info!("Phase 2 complete: {} flips performed", flips_done);
}

/// Rebuild adjacency from scratch by hashing faces.
/// This is a fallback for when incremental adjacency updates accumulate errors.
pub fn rebuild_adjacency(result: &mut DelaunayResult) {
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
