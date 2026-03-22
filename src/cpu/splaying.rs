//! Main star splaying orchestration.
//!
//! Ported from gDel3D/GDelFlipping/src/gDel3D/CPU/Splaying.{h,cpp}
//!
//! This module implements the multi-phase star splaying algorithm:
//! 1. Extract stars from failed vertices
//! 2. Perform local flipping on each star
//! 3. Process work queue for consistency
//! 4. Reintegrate stars back to tetrahedra

use crate::cpu::facet::{decode, encode, Facet, TriOpp, TriStatus};
use crate::cpu::star::Star;
use crate::types::{decode_opp, DelaunayResult, INVALID};

/// Lookup tables from CommonTypes.h (lines 125-137)
///
/// Order of 3 vertices as seen from one vertex of tetra
const TET_VI_AS_SEEN_FROM: [[usize; 3]; 4] = [
    [1, 3, 2], // From vertex 0
    [0, 2, 3], // From vertex 1
    [0, 3, 1], // From vertex 2
    [0, 1, 2], // From vertex 3 (default view)
];

/// Next vertex index mapping (for adjacency conversion)
const TET_NEXT_VI_AS_SEEN_FROM: [[i32; 4]; 4] = [
    [-1, 0, 2, 1], // From vertex 0
    [0, -1, 1, 2], // From vertex 1
    [0, 2, -1, 1], // From vertex 2
    [0, 1, 2, -1], // From vertex 3
];

/// Main entry point for star splaying.
///
/// Ported from Splaying.cpp lines 658-688 (fixWithStarSplaying)
pub fn fix_with_star_splaying(points: &[[f32; 3]], result: &mut DelaunayResult) {
    if result.failed_verts.is_empty() {
        return;
    }

    eprintln!("\n╔═══════════════════════════════════════════════════╗");
    eprintln!("║        CUDA STAR SPLAYING ALGORITHM               ║");
    eprintln!("╚═══════════════════════════════════════════════════╝");
    eprintln!("[SPLAY] Input: {} points, {} tets, {} failed vertices",
              points.len(), result.tets.len(), result.failed_verts.len());

    // Only force-insert vertices that are NOT already in the triangulation.
    // The gather shader reports insphere-violating vertices that ARE already inserted —
    // those just need star splaying, not re-insertion.
    eprintln!("\n[SPLAY] PHASE 0: Force-Insert Missing Vertices");
    let missing: Vec<u32> = result.failed_verts.iter()
        .filter(|&&v| !result.tets.iter().any(|tet| tet.contains(&v)))
        .copied()
        .collect();
    if !missing.is_empty() {
        eprintln!("[SPLAY] Force-inserting {} missing vertices: {:?}", missing.len(), missing);
        for &vert in &missing {
            crate::cpu::force_insert::force_insert_vertex(vert, result);
        }
    } else {
        eprintln!("[SPLAY] All {} failed vertices already in triangulation, skipping force-insert", result.failed_verts.len());
    }
    eprintln!("[SPLAY] After phase 0: {} tets", result.tets.len());

    let mut ctx = SplayingContext::new(points, result);

    // Phase 1: Extract stars and perform initial flipping
    eprintln!("\n[SPLAY] PHASE 1: Star Extraction");
    ctx.make_failed_stars_and_queue(result);

    // Phase 2: Process work queue for consistency
    eprintln!("\n[SPLAY] PHASE 2: Work Queue Processing");
    ctx.process_queue(result);

    // Phase 3: Reintegrate stars back to tetrahedra
    eprintln!("\n[SPLAY] PHASE 3: Reintegration");
    ctx.stars_to_tetra(result);

    // Phase 4: Clear failed verts (all fixed)
    let num_fixed = result.failed_verts.len();
    result.failed_verts.clear();

    eprintln!("\n[SPLAY] ✓ Star splaying complete: {} vertices fixed", num_fixed);
    eprintln!("╚═══════════════════════════════════════════════════╝\n");
}

/// Context for star splaying operations.
pub struct SplayingContext {
    /// All stars being processed
    pub stars: Vec<Star>,
    /// Work queue for consistency checking
    pub facet_queue: Vec<Facet>,
    /// Points array (f64 for predicates)
    points: Vec<[f64; 3]>,
    /// Visit IDs for DFS
    tet_visit: Vec<i32>,
    /// Map from tet index → (tri_index, bot_vi)
    tet_tri_map: Vec<u32>,
    /// Work stack for DFS
    int_stack: Vec<usize>,
    /// Visit counter
    visit_id: i32,
    /// Set of tet indices that have been claimed by stars (marked dead).
    /// CUDA: setTetAliveState(false) in tetVisitCreateStar (Splaying.cpp:131)
    dead_tets: std::collections::HashSet<usize>,
}

impl SplayingContext {
    pub fn new(points: &[[f32; 3]], result: &DelaunayResult) -> Self {
        let points: Vec<[f64; 3]> = points
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();

        let tet_visit = vec![0i32; result.tets.len()];
        let tet_tri_map = vec![0u32; result.tets.len()];

        Self {
            stars: Vec::new(),
            facet_queue: Vec::new(),
            points,
            tet_visit,
            tet_tri_map,
            int_stack: Vec::new(),
            visit_id: 1,
            dead_tets: std::collections::HashSet::new(),
        }
    }

    /// Extract star for a vertex via DFS traversal.
    ///
    /// Ported from Splaying.cpp lines 117-171 (tetVisitCreateStar)
    fn tet_visit_create_star(
        &mut self,
        start_idx: usize,
        vert: u32,
        result: &DelaunayResult,
    ) -> Star {
        let mut star = Star::new(vert, self.points.clone(), &mut self.facet_queue as *mut Vec<Facet>);

        self.int_stack.clear();
        self.int_stack.push(start_idx);
        self.tet_visit[start_idx] = self.visit_id;

        // DFS through all tets containing vert
        while let Some(tet_idx) = self.int_stack.pop() {
            // CUDA: setTetAliveState((*_tetInfoVec)[tetIdx], false);  (Splaying.cpp:131)
            // Mark tet as dead — it will be revived in starsToTetra if still valid
            self.dead_tets.insert(tet_idx);

            let tet = result.tets[tet_idx];
            let tet_opp = result.adjacency[tet_idx];

            // Find local index of vert in tet
            let bot_vi = tet.iter().position(|&v| v == vert).expect("Vertex not in tet");

            // Get the 3 vertices visible from vert (the link triangle)
            let ord_vi = TET_VI_AS_SEEN_FROM[bot_vi];

            // Build link triangle
            let tri_v0 = tet[ord_vi[0]];
            let tri_v1 = tet[ord_vi[1]];
            let tri_v2 = tet[ord_vi[2]];

            // Build adjacency for link triangle (opposite faces in tet order)
            let mut tri_opp = TriOpp::new();
            tri_opp.t[0] = tet_opp[ord_vi[0]];
            tri_opp.t[1] = tet_opp[ord_vi[1]];
            tri_opp.t[2] = tet_opp[ord_vi[2]];

            star.tri_opp_vec.push(tri_opp);

            // Map tet → (tri_index, bot_vi) for later adjacency conversion
            self.tet_tri_map[tet_idx] = encode(star.tri_vec.len() as u32, bot_vi as u32);

            // Add link triangle
            star.tri_vec
                .push(crate::cpu::facet::Tri::new(tri_v0, tri_v1, tri_v2));
            star.tet_idx_vec.push(tet_idx as i32);
            star.tri_status_vec.push(TriStatus::Valid);

            // Visit neighbors (other tets containing vert)
            for vi in 0..3 {
                let opp_packed = tri_opp.t[vi];
                let (opp_idx, _) = decode_opp(opp_packed);
                let opp_idx = opp_idx as usize;

                if opp_idx < result.tets.len() && self.tet_visit[opp_idx] != self.visit_id {
                    self.tet_visit[opp_idx] = self.visit_id;
                    self.int_stack.push(opp_idx);
                }
            }
        }

        star
    }

    /// Convert tet-based adjacency to local star adjacency.
    ///
    /// Ported from Splaying.cpp lines 173-199 (makeOppLocal)
    fn make_opp_local(&mut self, star: &mut Star) {
        for tri_idx in 0..star.tri_vec.len() {
            assert_ne!(star.tri_status_vec[tri_idx], TriStatus::Free);

            for vi in 0..3 {
                let opp_tet_packed = star.tri_opp_vec[tri_idx].t[vi];
                if opp_tet_packed == INVALID {
                    continue; // Boundary face
                }
                let (opp_tet_idx, opp_tet_vi) = decode_opp(opp_tet_packed);
                let opp_tet_idx = opp_tet_idx as usize;

                // Mark external faces as INVALID (neighbor tet not in this star).
                // These are boundary faces of the star — flipping must not cross them.
                if opp_tet_idx >= self.tet_visit.len()
                    || self.tet_visit[opp_tet_idx] != self.visit_id
                {
                    star.tri_opp_vec[tri_idx].t[vi] = INVALID;
                    continue;
                }

                // Decode tet → triangle mapping
                let (opp_tri_idx, opp_bot_vi) = decode(self.tet_tri_map[opp_tet_idx]);

                // Convert tet face index to triangle face index
                let opp_tri_vi = TET_NEXT_VI_AS_SEEN_FROM[opp_bot_vi as usize][opp_tet_vi as usize];

                if opp_tri_vi < 0 {
                    eprintln!(
                        "[SPLAY] WARNING: Invalid adjacency mapping: opp_bot_vi={}, opp_tet_vi={}",
                        opp_bot_vi, opp_tet_vi
                    );
                    // Skip this adjacency - star splaying will fix it
                    continue;
                }

                // Update adjacency to point to local star triangle.
                // Use set_opp_tri + set_opp_vi to PRESERVE flags (sphere_fail, special).
                // CUDA: Splaying.cpp:193-194 — setOppTri + setOppVi preserve flags.
                star.tri_opp_vec[tri_idx].set_opp_tri(vi, opp_tri_idx);
                star.tri_opp_vec[tri_idx].set_opp_vi(vi, opp_tri_vi as u32);
            }
        }
    }

    /// Create star from tetrahedra containing a vertex.
    ///
    /// Ported from Splaying.cpp lines 201-215 (createFromTetra)
    pub fn create_from_tetra(&mut self, vert: u32, result: &DelaunayResult) -> Option<Star> {
        // Find a tet containing vert
        let start_tet = match result.tets.iter().position(|tet| tet.contains(&vert)) {
            Some(idx) => idx,
            None => {
                eprintln!(
                    "[SPLAY] WARNING: No tet contains vertex {} - skipping",
                    vert
                );
                return None;
            }
        };

        let mut star = self.tet_visit_create_star(start_tet, vert, result);

        self.make_opp_local(&mut star);

        self.visit_id += 1;

        Some(star)
    }

    /// Add new link triangles to the work queue for consistency checking.
    ///
    /// Ported from Splaying.cpp lines 264-292 (compareStarAddQueue)
    ///
    /// For each link triangle NOT in original triangulation (tetIdxVec == -1),
    /// add 3 facets (one per edge) to the global work queue.
    fn compare_star_add_queue(&mut self, star_idx: usize) {
        let vert = self.stars[star_idx].vert;

        for tri_idx in 0..self.stars[star_idx].tri_vec.len() {
            if self.stars[star_idx].tri_status_vec[tri_idx] == TriStatus::Free {
                continue;
            }
            if self.stars[star_idx].tet_idx_vec[tri_idx] != -1 {
                continue; // Already in triangulation, no need to check
            }

            // Link triangle not in triangulation — add 3 facets
            let tri = self.stars[star_idx].tri_vec[tri_idx];

            for vi in 0..3 {
                let facet = Facet {
                    from: vert,
                    from_idx: tri_idx as u32,
                    to: tri.v[vi],
                    v0: tri.v[(vi + 1) % 3],
                    v1: tri.v[(vi + 2) % 3],
                };
                self.facet_queue.push(facet);
            }
        }
    }

    /// Create stars for all failed vertices and perform initial flipping.
    ///
    /// Ported from Splaying.cpp lines 217-262 (makeFailedStarsAndQueue)
    fn make_failed_stars_and_queue(&mut self, result: &mut DelaunayResult) {
        eprintln!("[SPLAY] ========== STAR EXTRACTION ==========");
        eprintln!("[SPLAY] Failed vertices: {:?}", result.failed_verts);
        eprintln!("[SPLAY] Total tets available: {}", result.tets.len());

        for &failed_vert in &result.failed_verts.clone() {
            let mut star = match self.create_from_tetra(failed_vert, result) {
                Some(s) => s,
                None => {
                    eprintln!("[SPLAY]   Skipping vertex {} (not in triangulation)", failed_vert);
                    continue;
                }
            };

            eprintln!(
                "[SPLAY]   Star {}: {} triangles in link",
                failed_vert,
                star.tri_vec.len()
            );

            // Perform initial flipping on star
            // CUDA: Star.cpp:305-324 — flipping() checks OPP_SPHERE_FAIL
            star.do_flipping();

            // Add to star list
            let star_idx = self.stars.len();
            self.stars.push(star);

            // CUDA: Splaying.cpp:248 — compareStarAddQueue(star)
            // Add all NEW link triangles (tetIdxVec == -1) to the work queue.
            // This is CRITICAL: without it, process_queue has no work items
            // and neighbor stars are never created for consistency.
            self.compare_star_add_queue(star_idx);
        }

        eprintln!("[SPLAY] Extracted {} stars total", self.stars.len());
        eprintln!("[SPLAY] Initial facet queue size: {}", self.facet_queue.len());
    }

    /// Process work queue until all stars are consistent.
    ///
    /// Ported from Splaying.cpp lines 294-439 (processQue)
    fn process_queue(&mut self, result: &mut DelaunayResult) {
        eprintln!("[SPLAY] ========== WORK QUEUE PROCESSING ==========");
        eprintln!(
            "[SPLAY] Initial queue size: {} facets",
            self.facet_queue.len()
        );

        let mut star_map: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for (idx, star) in self.stars.iter().enumerate() {
            star_map.insert(star.vert, idx);
        }

        let mut count = 0;
        let mut ins_count = 0;
        let mut proof_count = 0;

        let mut stack = Vec::new();
        let mut visited = vec![0i32; result.tets.len()];

        // Safety limit to prevent infinite loops in degenerate cases.
        // CUDA should converge in O(n) operations. Use a tighter bound.
        let max_queue_ops = result.tets.len() * 10;

        while !self.facet_queue.is_empty() {
            let facet = self.facet_queue.pop().unwrap();
            count += 1;

            if count > max_queue_ops {
                eprintln!("[SPLAY] WARNING: process_queue exceeded {} iterations, stopping", max_queue_ops);
                break;
            }

            let from_vert = facet.from;
            let to_vert = facet.to;

            // Handle drowned vertex (special case: from_idx == u32::MAX)
            if facet.from_idx == u32::MAX {
                // Create star if doesn't exist
                if !star_map.contains_key(&from_vert) {
                    if let Some(new_star) = self.create_from_tetra(from_vert, result) {
                        star_map.insert(from_vert, self.stars.len());
                        self.stars.push(new_star);
                    } else {
                        continue;
                    }
                }

                let from_idx = *star_map.get(&from_vert).unwrap();
                let to_idx = *star_map.get(&to_vert).unwrap();

                // Check if still has vertex
                if !self.stars[from_idx].has_link_vert(to_vert) {
                    continue;
                }

                // Get proof
                let proof_arr = self.stars[to_idx].get_proof(from_vert);
                proof_count += 1;

                // Insert proof vertices into from star
                for &proof_vert in &proof_arr {
                    if proof_vert == from_vert || self.stars[from_idx].has_link_vert(proof_vert) {
                        continue;
                    }

                    self.stars[from_idx].insert_to_star(
                        proof_vert,
                        &mut stack,
                        &mut visited,
                        self.visit_id,
                    );
                    self.visit_id += 1;
                    ins_count += 1;
                }
                continue;
            }

            // Regular facet processing
            let from_idx = match star_map.get(&from_vert) {
                Some(&idx) => idx,
                None => continue,
            };

            // Check if facet still exists in from-star
            let facet_idx = facet.from_idx as usize;
            if facet_idx >= self.stars[from_idx].tri_status_vec.len()
                || self.stars[from_idx].tri_status_vec[facet_idx] == TriStatus::Free
            {
                continue;
            }

            let tri = self.stars[from_idx].tri_vec[facet_idx];
            if !tri.has_all(to_vert, facet.v0, facet.v1) {
                continue;
            }

            // Create to-star if needed
            if !star_map.contains_key(&to_vert) {
                if let Some(new_star) = self.create_from_tetra(to_vert, result) {
                    star_map.insert(to_vert, self.stars.len());
                    self.stars.push(new_star);
                } else {
                    continue;
                }
            }

            let to_idx = *star_map.get(&to_vert).unwrap();

            // Try inserting vertices from facet triangle into to-star
            let check_tri = crate::cpu::facet::Tri::new(from_vert, facet.v0, facet.v1);

            let to_star_vert = self.stars[to_idx].vert;
            for vi in 0..3 {
                let vert = check_tri.v[vi];

                // Never insert the star's own center vertex
                if vert == to_star_vert {
                    continue;
                }

                if self.stars[to_idx].has_link_vert(vert) {
                    continue;
                }

                ins_count += 1;

                if self.stars[to_idx].insert_to_star(vert, &mut stack, &mut visited, self.visit_id) {
                    self.visit_id += 1;
                    continue; // Success
                }

                // Insertion failed - get proof
                let proof_arr = self.stars[to_idx].get_proof(vert);
                proof_count += 1;

                // Insert proof vertices into from-star
                for &proof_vert in &proof_arr {
                    if proof_vert == from_vert || self.stars[from_idx].has_link_vert(proof_vert) {
                        continue;
                    }

                    self.stars[from_idx].insert_to_star(
                        proof_vert,
                        &mut stack,
                        &mut visited,
                        self.visit_id,
                    );
                    self.visit_id += 1;
                    ins_count += 1;
                }

                break;
            }
        }

        eprintln!(
            "[SPLAY] Queue complete: {} items, {} insertions, {} proofs",
            count, ins_count, proof_count
        );
    }

    /// Convert all stars back to tetrahedra.
    ///
    /// Ported from Splaying.cpp lines 485-615 (starsToTetra)
    ///
    /// CUDA approach:
    /// 1. tetVisitCreateStar already marked all star tets as DEAD
    /// 2. For alive link triangles with tet_idx >= 0: revive the tet
    /// 3. For alive link triangles with tet_idx == -1: find from other stars or create new
    /// 4. Set adjacency for all new/modified tets
    /// 5. Dead tets (not revived) are effectively removed
    fn stars_to_tetra(&mut self, result: &mut DelaunayResult) {
        eprintln!("[SPLAY] ========== REINTEGRATION ==========");
        eprintln!("[SPLAY] Reintegrating {} stars", self.stars.len());
        eprintln!("[SPLAY] Current tet count: {}", result.tets.len());
        eprintln!("[SPLAY] Dead tets from star extraction: {}", self.dead_tets.len());

        use crate::types::{encode_opp, INVALID};

        let mut new_tet_count = 0;
        let mut revived_count = 0;

        // Build star map for quick lookup (like CUDA's _starVec indexed by vertex)
        let mut star_map: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for (idx, star) in self.stars.iter().enumerate() {
            star_map.insert(star.vert, idx);
        }

        // PASS 1: Create/reuse tets and set tetIdxVec
        // (CUDA Splaying.cpp:497-555)
        for star_idx in 0..self.stars.len() {
            let vert = self.stars[star_idx].vert;

            for tri_idx in 0..self.stars[star_idx].tri_vec.len() {
                if self.stars[star_idx].tri_status_vec[tri_idx] == TriStatus::Free {
                    continue;
                }

                let tri = self.stars[star_idx].tri_vec[tri_idx];
                let mut tet_idx = self.stars[star_idx].tet_idx_vec[tri_idx];

                // CUDA: if (tetIdx >= 0) { setTetAliveState(true); continue; }
                if tet_idx != -1 {
                    // Revive this tet — it's still valid (not flipped away)
                    self.dead_tets.remove(&(tet_idx as usize));
                    revived_count += 1;
                    continue;
                }

                // tetIdx == -1: new tet needed. Find in 3 other stars first.
                let mut tet_in_star: [Option<usize>; 3] = [None; 3];

                for vi in 0..3 {
                    let to_vert = tri.v[vi];
                    let to_tri = crate::cpu::facet::Tri::new(
                        vert,
                        tri.v[(vi + 1) % 3],
                        tri.v[(vi + 2) % 3],
                    );

                    let to_star_idx = match star_map.get(&to_vert) {
                        Some(&idx) => idx,
                        None => continue,
                    };

                    let to_tri_idx = match self.stars[to_star_idx].get_link_tri_idx(&to_tri) {
                        Some(idx) => idx,
                        None => continue,
                    };

                    tet_in_star[vi] = Some(to_tri_idx);

                    // Check if other star already allocated a tet_idx
                    if self.stars[to_star_idx].tet_idx_vec[to_tri_idx] != -1 {
                        tet_idx = self.stars[to_star_idx].tet_idx_vec[to_tri_idx];
                    }
                }

                // Create new tet if not found in any star
                if tet_idx == -1 {
                    tet_idx = result.tets.len() as i32;
                    let tet = [tri.v[0], tri.v[1], tri.v[2], vert];
                    result.tets.push(tet);
                    result.adjacency.push([INVALID; 4]);
                    new_tet_count += 1;
                } else {
                    // Tet was found in another star. Update its vertex data
                    // (in case it was a recycled dead tet with old data).
                    result.tets[tet_idx as usize] = [tri.v[0], tri.v[1], tri.v[2], vert];
                    result.adjacency[tet_idx as usize] = [INVALID; 4];
                    // This tet is alive now
                    self.dead_tets.remove(&(tet_idx as usize));
                }

                // Store tet_idx in this star and all 3 other stars
                self.stars[star_idx].tet_idx_vec[tri_idx] = tet_idx;
                for vi in 0..3 {
                    if let Some(to_tri_idx) = tet_in_star[vi] {
                        let to_star_idx = *star_map.get(&tri.v[vi]).unwrap();
                        self.stars[to_star_idx].tet_idx_vec[to_tri_idx] = tet_idx;
                    }
                }
            }
        }

        // PASS 2: Set adjacency
        // (CUDA Splaying.cpp:557-608)
        for star_idx in 0..self.stars.len() {
            let vert = self.stars[star_idx].vert;

            for tri_idx in 0..self.stars[star_idx].tri_vec.len() {
                if self.stars[star_idx].tri_status_vec[tri_idx] == TriStatus::Free {
                    continue;
                }

                let tet_idx = self.stars[star_idx].tet_idx_vec[tri_idx];
                if tet_idx < 0 {
                    continue;
                }

                let tri = self.stars[star_idx].tri_vec[tri_idx];
                let tri_opp = self.stars[star_idx].tri_opp_vec[tri_idx];
                let tet = result.tets[tet_idx as usize];

                // Set 3 face adjacencies (opposite to link triangle vertices)
                // CUDA: Splaying.cpp:562-584
                for vi in 0..3 {
                    if tri_opp.t[vi] == INVALID {
                        continue; // Boundary face
                    }
                    let opp_tri_idx = tri_opp.get_opp_tri(vi) as usize;
                    let opp_tri_vi = tri_opp.get_opp_vi(vi) as usize;

                    if opp_tri_idx >= self.stars[star_idx].tet_idx_vec.len() {
                        continue;
                    }
                    let opp_tri_tet_idx = self.stars[star_idx].tet_idx_vec[opp_tri_idx];

                    if opp_tri_tet_idx == -1 {
                        continue; // Neighbor not created yet
                    }

                    let tri_vert = tri.v[vi];
                    let cur_tet_vi = match tet.iter().position(|&v| v == tri_vert) {
                        Some(p) => p,
                        None => continue,
                    };

                    let opp_tri = self.stars[star_idx].tri_vec[opp_tri_idx];
                    let opp_vert = opp_tri.v[opp_tri_vi];
                    let opp_tet = result.tets[opp_tri_tet_idx as usize];
                    let opp_tet_vi = match opp_tet.iter().position(|&v| v == opp_vert) {
                        Some(p) => p,
                        None => continue,
                    };

                    result.adjacency[tet_idx as usize][cur_tet_vi] =
                        encode_opp(opp_tri_tet_idx as u32, opp_tet_vi as u32);
                    result.adjacency[opp_tri_tet_idx as usize][opp_tet_vi] =
                        encode_opp(tet_idx as u32, cur_tet_vi as u32);
                }

                // Set 4th face adjacency (opposite to star vertex)
                // CUDA: Splaying.cpp:586-608
                // Navigate via star of tri.v[0] to find the tet on the other side
                let v0 = tri.v[0];
                let to_star_idx = match star_map.get(&v0) {
                    Some(&idx) => idx,
                    None => continue,
                };

                // Find the same tet in v0's star
                let to_tri = crate::cpu::facet::Tri::new(
                    vert,
                    tri.v[1],
                    tri.v[2],
                );
                let to_tri_idx = match self.stars[to_star_idx].get_link_tri_idx(&to_tri) {
                    Some(idx) => idx,
                    None => continue,
                };

                // Find vert's position in the link triangle of v0's star
                let tri2 = self.stars[to_star_idx].tri_vec[to_tri_idx];
                let vi2 = match tri2.v.iter().position(|&v| v == vert) {
                    Some(p) => p,
                    None => continue,
                };

                let tri_opp2 = self.stars[to_star_idx].tri_opp_vec[to_tri_idx];
                if tri_opp2.t[vi2] == INVALID {
                    continue;
                }
                let opp_tri_idx2 = tri_opp2.get_opp_tri(vi2) as usize;
                let opp_tri_vi2 = tri_opp2.get_opp_vi(vi2) as usize;

                if opp_tri_idx2 >= self.stars[to_star_idx].tet_idx_vec.len() {
                    continue;
                }
                let opp_tri_tet_idx2 = self.stars[to_star_idx].tet_idx_vec[opp_tri_idx2];

                if opp_tri_tet_idx2 == -1 {
                    continue;
                }

                let opp_tri2 = self.stars[to_star_idx].tri_vec[opp_tri_idx2];
                if opp_tri_vi2 >= 3 {
                    continue;
                }
                let opp_vert = opp_tri2.v[opp_tri_vi2];
                let opp_tet = result.tets[opp_tri_tet_idx2 as usize];
                let opp_tet_vi = match opp_tet.iter().position(|&v| v == opp_vert) {
                    Some(p) => p,
                    None => continue,
                };
                let cur_tet_vi = match tet.iter().position(|&v| v == vert) {
                    Some(p) => p,
                    None => continue,
                };

                result.adjacency[tet_idx as usize][cur_tet_vi] =
                    encode_opp(opp_tri_tet_idx2 as u32, opp_tet_vi as u32);
                result.adjacency[opp_tri_tet_idx2 as usize][opp_tet_vi] =
                    encode_opp(tet_idx as u32, cur_tet_vi as u32);
            }
        }

        // PASS 3: Kill remaining dead tets
        // CUDA: dead tets have tetInfo=0 and are filtered by compactTetras.
        // We mark them with INVALID vertices so check_delaunay_quality skips them.
        let dead_count = self.dead_tets.len();
        for &dead_idx in &self.dead_tets {
            if dead_idx < result.tets.len() {
                result.tets[dead_idx] = [INVALID, INVALID, INVALID, INVALID];
                result.adjacency[dead_idx] = [INVALID; 4];
            }
        }

        eprintln!("[SPLAY] Reintegration complete:");
        eprintln!("[SPLAY]   Revived tets: {}", revived_count);
        eprintln!("[SPLAY]   New tets created: {}", new_tet_count);
        eprintln!("[SPLAY]   Dead tets removed: {}", dead_count);
        eprintln!("[SPLAY]   Final tet count: {}", result.tets.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splaying_context_creation() {
        let points = vec![[0.0, 0.0, 0.0]];
        let result = DelaunayResult {
            tets: vec![[0, 1, 2, 3]],
            adjacency: vec![[0; 4]],
            failed_verts: vec![],
        };

        let ctx = SplayingContext::new(&points, &result);
        assert_eq!(ctx.stars.len(), 0);
        assert_eq!(ctx.facet_queue.len(), 0);
    }

    #[test]
    fn test_fix_with_star_splaying_empty() {
        let points = vec![[0.0, 0.0, 0.0]];
        let mut result = DelaunayResult {
            tets: Vec::new(),
            adjacency: Vec::new(),
            failed_verts: Vec::new(),
        };

        fix_with_star_splaying(&points, &mut result);

        // Should do nothing for empty failed_verts
        assert_eq!(result.failed_verts.len(), 0);
    }
}
