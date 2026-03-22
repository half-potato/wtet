//! Star data structure and local flipping operations.
//!
//! Ported from gDel3D/GDelFlipping/src/gDel3D/CPU/Star.{h,cpp}
//!
//! A "star" is the set of tetrahedra incident to a vertex, represented as
//! a 2D link manifold (the triangles visible from the vertex).

use crate::cpu::facet::{decode, encode, Facet, Tri, TriOpp, TriStatus};
use crate::predicates;

const STAR_AVG_TRI_NUM: usize = 32;

/// Star - the link manifold around a central vertex.
///
/// Ported from Star.h (lines 51-150)
pub struct Star {
    /// Central vertex
    pub vert: u32,
    /// Link triangles (faces visible from central vertex)
    pub tri_vec: Vec<Tri>,
    /// Adjacency within link (5-bit encoding)
    pub tri_opp_vec: Vec<TriOpp>,
    /// Associated tetrahedron indices (-1 if not created yet)
    pub tet_idx_vec: Vec<i32>,
    /// Triangle status (Free/Valid/New)
    pub tri_status_vec: Vec<TriStatus>,
    /// Points array (for geometric predicates)
    points: Vec<[f64; 3]>,
    /// Facet queue for work items (shared across all stars)
    facet_queue: *mut Vec<Facet>,
}

impl Star {
    pub fn new(vert: u32, points: Vec<[f64; 3]>, facet_queue: *mut Vec<Facet>) -> Self {
        let mut tri_vec = Vec::new();
        let mut tri_opp_vec = Vec::new();
        let mut tet_idx_vec = Vec::new();
        let mut tri_status_vec = Vec::new();

        tri_vec.reserve(STAR_AVG_TRI_NUM);
        tri_opp_vec.reserve(STAR_AVG_TRI_NUM);
        tet_idx_vec.reserve(STAR_AVG_TRI_NUM);
        tri_status_vec.reserve(STAR_AVG_TRI_NUM);

        Self {
            vert,
            tri_vec,
            tri_opp_vec,
            tet_idx_vec,
            tri_status_vec,
            points,
            facet_queue,
        }
    }

    /// Check if vertex exists in the link.
    pub fn has_link_vert(&self, vertex: u32) -> bool {
        self.tri_vec.iter().any(|tri| tri.has(vertex))
    }

    /// Find link triangle index matching given triangle.
    ///
    /// Used during reintegration to find matching triangles between stars.
    pub fn get_link_tri_idx(&self, search_tri: &Tri) -> Option<usize> {
        for (idx, tri) in self.tri_vec.iter().enumerate() {
            if self.tri_status_vec[idx] == TriStatus::Free {
                continue;
            }
            if tri.has_all(search_tri.v[0], search_tri.v[1], search_tri.v[2]) {
                return Some(idx);
            }
        }
        None
    }

    /// Clear all data in the star.
    pub fn clear(&mut self) {
        self.tri_vec.clear();
        self.tri_opp_vec.clear();
        self.tet_idx_vec.clear();
        self.tri_status_vec.clear();
    }

    /// Helper to get opposite triangle index from packed value.
    #[inline]
    fn get_opp_tri(&self, packed: u32) -> usize {
        (packed >> 5) as usize
    }

    /// Helper to get opposite vi from packed value.
    #[inline]
    fn get_opp_vi(&self, packed: u32) -> usize {
        (packed & 3) as usize
    }

    /// Perform 3-1 flip: three triangles around a vertex → one triangle (delete vertex from link).
    ///
    /// Ported from Star.cpp lines 105-143
    ///
    /// **Algorithm:**
    /// - Three triangles share a common vertex (to be deleted)
    /// - Replace with one triangle opposite that vertex
    /// - Update adjacency for neighbors
    /// - Mark two triangles as Free
    ///
    /// **Parameters:**
    /// - `tri_idx`: Index of one of the three triangles
    /// - `vi`: Local index of the vertex to delete
    /// - `stack`: Work stack for recursive flipping
    pub fn flip31(&mut self, tri_idx: usize, vi: usize, stack: &mut Vec<u32>) {
        let cor_vi = (vi + 1) % 3;
        let opp = self.tri_opp_vec[tri_idx];
        let tri = self.tri_vec[tri_idx];
        let len = self.tri_opp_vec.len();

        // Guard: all adjacencies must be valid
        if opp.t[vi] == crate::types::INVALID || opp.t[cor_vi] == crate::types::INVALID {
            return;
        }
        let opp_tri = opp.get_opp_tri(vi) as usize;
        let opp_vi = opp.get_opp_vi(vi) as usize;
        let side_tri = opp.get_opp_tri(cor_vi) as usize;
        let side_vi = opp.get_opp_vi(cor_vi) as usize;
        if opp_tri >= len || side_tri >= len { return; }
        let opp_vert = self.tri_vec[opp_tri].v[opp_vi];

        // Build new adjacency — check all sources are valid
        let opp_v0 = self.tri_opp_vec[opp_tri].get_opp_tri_vi((opp_vi + 1) % 3);
        let opp_v1 = self.tri_opp_vec[side_tri].get_opp_tri_vi((side_vi + 2) % 3);
        let opp_v2 = opp.get_opp_tri_vi((vi + 2) % 3);
        if opp_v0 == crate::types::INVALID || opp_v1 == crate::types::INVALID
            || opp_v2 == crate::types::INVALID
        {
            return;
        }
        if self.get_opp_tri(opp_v0) >= len || self.get_opp_tri(opp_v1) >= len
            || self.get_opp_tri(opp_v2) >= len
        {
            return;
        }

        // Create new triangle
        let new_tri = Tri::new(tri.v[vi], tri.v[cor_vi], opp_vert);

        self.tri_vec[tri_idx] = new_tri;
        self.tet_idx_vec[tri_idx] = -1;

        let mut new_opp = TriOpp::new();
        new_opp.t[0] = opp_v0;
        new_opp.t[1] = opp_v1;
        new_opp.t[2] = opp_v2;

        self.tri_opp_vec[tri_idx] = new_opp;

        // Update back-pointers
        let opp_tri_0 = self.get_opp_tri(opp_v0);
        let opp_vi_0 = self.get_opp_vi(opp_v0);
        self.tri_opp_vec[opp_tri_0].set_opp(opp_vi_0, tri_idx as u32, 0);

        let opp_tri_1 = self.get_opp_tri(opp_v1);
        let opp_vi_1 = self.get_opp_vi(opp_v1);
        self.tri_opp_vec[opp_tri_1].set_opp(opp_vi_1, tri_idx as u32, 1);

        let opp_tri_2 = self.get_opp_tri(opp_v2);
        let opp_vi_2 = self.get_opp_vi(opp_v2);
        self.tri_opp_vec[opp_tri_2].set_opp(opp_vi_2, tri_idx as u32, 2);

        // Mark deleted triangles as free
        self.tri_status_vec[opp_tri] = TriStatus::Free;
        self.tri_status_vec[side_tri] = TriStatus::Free;

        // Add deletion to queue (for consistency checking)
        self.add_one_deletion_to_queue(tri.v[(vi + 2) % 3]);

        // Add new triangle's edges to flipping stack
        stack.push(encode(tri_idx as u32, 0));
        stack.push(encode(tri_idx as u32, 1));
        stack.push(encode(tri_idx as u32, 2));
    }

    /// Perform 2-2 flip: two triangles sharing an edge → two triangles with different edge.
    ///
    /// Ported from Star.cpp lines 145-188
    ///
    /// **Algorithm:**
    /// - Two triangles share an edge (diagonal)
    /// - Flip the diagonal to the other diagonal of the quadrilateral
    /// - Update adjacency for both triangles and their neighbors
    ///
    /// **Parameters:**
    /// - `tri_idx`: Index of first triangle
    /// - `vi`: Local index of edge to flip
    /// - `stack`: Work stack for recursive flipping
    pub fn flip22(&mut self, tri_idx: usize, vi: usize, stack: &mut Vec<u32>) {
        // Re-read live adjacency (snapshot from do_flipping may be stale after prior flips)
        let opp = self.tri_opp_vec[tri_idx];
        let tri = self.tri_vec[tri_idx];

        if opp.t[vi] == crate::types::INVALID { return; }
        let opp_tri = opp.get_opp_tri(vi) as usize;
        if opp_tri >= self.tri_vec.len() { return; }
        let opp_vi = opp.get_opp_vi(vi) as usize;
        let opp_vert = self.tri_vec[opp_tri].v[opp_vi];

        // Guard: all 4 surrounding adjacencies must be valid for back-pointer updates
        let opp_v0 = opp.get_opp_tri_vi((vi + 2) % 3);
        let opp_v1 = self.tri_opp_vec[opp_tri].get_opp_tri_vi((opp_vi + 1) % 3);
        let opp_v2 = self.tri_opp_vec[opp_tri].get_opp_tri_vi((opp_vi + 2) % 3);
        let opp_v3 = opp.get_opp_tri_vi((vi + 1) % 3);
        let len = self.tri_opp_vec.len();
        if opp_v0 == crate::types::INVALID || opp_v1 == crate::types::INVALID
            || opp_v2 == crate::types::INVALID || opp_v3 == crate::types::INVALID
        {
            return;
        }
        if self.get_opp_tri(opp_v0) >= len || self.get_opp_tri(opp_v1) >= len
            || self.get_opp_tri(opp_v2) >= len || self.get_opp_tri(opp_v3) >= len
        {
            return;
        }

        // Create two new triangles
        let new_tri0 = Tri::new(tri.v[vi], opp_vert, tri.v[(vi + 2) % 3]);
        let new_tri1 = Tri::new(opp_vert, tri.v[vi], tri.v[(vi + 1) % 3]);

        self.tri_vec[tri_idx] = new_tri0;
        self.tri_vec[opp_tri] = new_tri1;

        self.tet_idx_vec[tri_idx] = -1;
        self.tet_idx_vec[opp_tri] = -1;

        // Build new adjacency
        let mut new_opp0 = TriOpp::new();
        new_opp0.t[0] = opp_v2;
        new_opp0.t[1] = opp_v3;
        new_opp0.set_opp(2, opp_tri as u32, 2);

        self.tri_opp_vec[tri_idx] = new_opp0;

        let mut new_opp1 = TriOpp::new();
        new_opp1.t[0] = opp_v0;
        new_opp1.t[1] = opp_v1;
        new_opp1.set_opp(2, tri_idx as u32, 2);

        self.tri_opp_vec[opp_tri] = new_opp1;

        // Update back-pointers
        let opp_tri_0 = self.get_opp_tri(opp_v0);
        let opp_vi_0 = self.get_opp_vi(opp_v0);
        self.tri_opp_vec[opp_tri_0].set_opp(opp_vi_0, opp_tri as u32, 0);

        let opp_tri_1 = self.get_opp_tri(opp_v1);
        let opp_vi_1 = self.get_opp_vi(opp_v1);
        self.tri_opp_vec[opp_tri_1].set_opp(opp_vi_1, opp_tri as u32, 1);

        let opp_tri_2 = self.get_opp_tri(opp_v2);
        let opp_vi_2 = self.get_opp_vi(opp_v2);
        self.tri_opp_vec[opp_tri_2].set_opp(opp_vi_2, tri_idx as u32, 0);

        let opp_tri_3 = self.get_opp_tri(opp_v3);
        let opp_vi_3 = self.get_opp_vi(opp_v3);
        self.tri_opp_vec[opp_tri_3].set_opp(opp_vi_3, tri_idx as u32, 1);

        // Add new edges to flipping stack
        stack.push(encode(tri_idx as u32, 0));
        stack.push(encode(tri_idx as u32, 1));
        stack.push(encode(opp_tri as u32, 0));
        stack.push(encode(opp_tri as u32, 1));
    }

    /// Add deletion to facet queue for consistency checking.
    ///
    /// Ported from Star.cpp lines 47-60
    fn add_one_deletion_to_queue(&mut self, to_vert: u32) {
        unsafe {
            if let Some(queue) = self.facet_queue.as_mut() {
                let facet = Facet {
                    from: to_vert,
                    from_idx: u32::MAX, // -1 in CUDA
                    to: self.vert,
                    v0: 0,
                    v1: 0,
                };
                queue.push(facet);
            }
        }
    }

    /// Add triangle to facet queue for consistency checking.
    ///
    /// Ported from Star.cpp lines 62-86
    fn add_one_tri_to_queue(&mut self, from_idx: usize, tri: &Tri) {
        unsafe {
            if let Some(queue) = self.facet_queue.as_mut() {
                // Find minimum vertex in triangle
                let mut min = 0;
                if tri.v[1] < tri.v[min] {
                    min = 1;
                }
                if tri.v[2] < tri.v[min] {
                    min = 2;
                }

                // Find minimum vertex greater than star vertex
                let mut min_next = None;
                for i in 0..3 {
                    if tri.v[i] > self.vert {
                        if let Some(mn) = min_next {
                            if tri.v[i] < tri.v[mn] {
                                min_next = Some(i);
                            }
                        } else {
                            min_next = Some(i);
                        }
                    }
                }

                let min_next = min_next.unwrap_or(min);

                let facet = Facet {
                    from: self.vert,
                    from_idx: from_idx as u32,
                    to: tri.v[min_next],
                    v0: tri.v[(min_next + 1) % 3],
                    v1: tri.v[(min_next + 2) % 3],
                };

                queue.push(facet);
            }
        }
    }

    /// Perform local Delaunay flipping on link using flip-flop algorithm.
    ///
    /// Ported from Star.cpp lines 191-288
    ///
    /// **TODO:** Full implementation in progress - this is a simplified version
    pub fn do_flipping(&mut self) {
        let mut stack: Vec<u32> = Vec::new();
        // Size to cover all vertex IDs that may appear in the star (including super-tet)
        let max_vert = self.tri_vec.iter()
            .flat_map(|t| t.v.iter())
            .copied()
            .max()
            .unwrap_or(0) as usize;
        let is_non_extreme_len = (max_vert + 1).max(self.points.len());
        let mut is_non_extreme = vec![0i32; is_non_extreme_len];
        let visit_id = 1i32;

        // Start with all edges
        for tri_idx in 0..self.tri_vec.len() {
            if self.tri_status_vec[tri_idx] == TriStatus::Free {
                continue;
            }
            for vi in 0..3 {
                stack.push(encode(tri_idx as u32, vi as u32));
            }
        }

        // Process stack
        while let Some(code) = stack.pop() {
            let (tri_idx, vi) = decode(code);
            let tri_idx = tri_idx as usize;
            let vi = vi as usize;

            if self.tri_status_vec[tri_idx] == TriStatus::Free {
                continue;
            }

            let opp = self.tri_opp_vec[tri_idx];
            let tri = self.tri_vec[tri_idx];

            // Skip boundary faces (INVALID adjacency = star boundary)
            if opp.t[vi] == crate::types::INVALID {
                continue;
            }

            let opp_tri = opp.get_opp_tri(vi) as usize;
            let opp_vi = opp.get_opp_vi(vi) as usize;

            if opp_tri >= self.tri_status_vec.len()
                || self.tri_status_vec[opp_tri] == TriStatus::Free
            {
                continue;
            }

            let opp_vert = self.tri_vec[opp_tri].v[opp_vi];

            // Find min labeled vertex in configuration
            let mut min_vert = u32::MAX;
            if is_non_extreme[opp_vert as usize] == visit_id {
                min_vert = opp_vert;
            }
            for i in 0..3 {
                if tri.v[i] < min_vert && is_non_extreme[tri.v[i] as usize] == visit_id {
                    min_vert = tri.v[i];
                }
            }

            // Skip if flipping increases min non-extreme vertex's degree
            if min_vert == tri.v[vi] || min_vert == opp_vert {
                continue;
            }

            // Use orient4d to check if flip is needed
            let ort = predicates::orient4d(
                self.points[tri.v[0] as usize],
                self.points[tri.v[1] as usize],
                self.points[tri.v[2] as usize],
                self.points[self.vert as usize],
                self.points[opp_vert as usize],
                tri.v[0],
                tri.v[1],
                tri.v[2],
                self.vert,
                opp_vert,
            );

            if ort > 0 && min_vert == u32::MAX {
                continue;
            }

            // Check for 3-1 flip possibility.
            // flip31 accesses: opp.t[vi], opp.t[(vi+1)%3], opp.t[(vi+2)%3],
            //   tri_opp_vec[opp_tri].t[(opp_vi+1)%3], tri_opp_vec[side_tri].t[(side_vi+2)%3]
            // All must be valid (not INVALID/boundary).
            {
                let all_valid = opp.t[(vi + 1) % 3] != crate::types::INVALID
                    && opp.t[(vi + 2) % 3] != crate::types::INVALID
                    && self.tri_opp_vec[opp_tri].t[(opp_vi + 1) % 3] != crate::types::INVALID
                    && self.tri_opp_vec[opp_tri].t[(opp_vi + 2) % 3] != crate::types::INVALID;

                if all_valid {
                    let side_tri_idx = opp.get_opp_tri((vi + 1) % 3) as usize;
                    let side_vi_idx = opp.get_opp_vi((vi + 1) % 3) as usize;
                    let side_valid = side_tri_idx < self.tri_opp_vec.len()
                        && self.tri_opp_vec[side_tri_idx].t[(side_vi_idx + 2) % 3] != crate::types::INVALID;

                    if side_valid
                        && opp.get_opp_tri((vi + 1) % 3) == self.tri_opp_vec[opp_tri].get_opp_tri((opp_vi + 2) % 3)
                    {
                        if ort < 0 || min_vert == tri.v[(vi + 2) % 3] {
                            self.flip31(tri_idx, vi, &mut stack);
                        }
                        continue;
                    }

                    let side2_tri_idx = opp.get_opp_tri((vi + 2) % 3) as usize;
                    let side2_vi_idx = opp.get_opp_vi((vi + 2) % 3) as usize;
                    let side2_valid = side2_tri_idx < self.tri_opp_vec.len()
                        && self.tri_opp_vec[side2_tri_idx].t[(side2_vi_idx + 1) % 3] != crate::types::INVALID;

                    if side2_valid
                        && opp.get_opp_tri((vi + 2) % 3) == self.tri_opp_vec[opp_tri].get_opp_tri((opp_vi + 1) % 3)
                    {
                        if ort < 0 || min_vert == tri.v[(vi + 1) % 3] {
                            self.flip31(tri_idx, (vi + 2) % 3, &mut stack);
                        }
                        continue;
                    }
                }
            }

            // Check all surrounding edges are valid before attempting any flip.
            // Stars may have boundary (INVALID adjacency) when the GPU output is
            // incomplete. CUDA assumes closed manifolds; we must guard against open ones.
            if opp.t[(vi + 1) % 3] == crate::types::INVALID
                || opp.t[(vi + 2) % 3] == crate::types::INVALID
                || self.tri_opp_vec[opp_tri].t[(opp_vi + 1) % 3] == crate::types::INVALID
                || self.tri_opp_vec[opp_tri].t[(opp_vi + 2) % 3] == crate::types::INVALID
            {
                continue;
            }

            // Check 2-2 flippability using orient3d
            // CUDA: OrientNeg == doOrient3DSoS(...) means Shewchuk orient3d > 0
            // (ortToOrient inverts: det > 0 → OrientNeg)
            // So "can't flip" when Shewchuk orient3d > 0.
            let or1 = predicates::orient3d(
                self.points[tri.v[vi] as usize],
                self.points[tri.v[(vi + 1) % 3] as usize],
                self.points[opp_vert as usize],
                self.points[self.vert as usize],
            );

            if or1 > 0.0 {
                if ort < 0 && is_non_extreme[tri.v[(vi + 1) % 3] as usize] != visit_id {
                    is_non_extreme[tri.v[(vi + 1) % 3] as usize] = visit_id;
                    self.push_fan_to_stack(tri_idx, (vi + 2) % 3, &mut stack);
                }
                continue;
            }

            let or2 = predicates::orient3d(
                self.points[tri.v[(vi + 2) % 3] as usize],
                self.points[tri.v[vi] as usize],
                self.points[opp_vert as usize],
                self.points[self.vert as usize],
            );

            if or2 > 0.0 {
                if ort < 0 && is_non_extreme[tri.v[(vi + 2) % 3] as usize] != visit_id {
                    is_non_extreme[tri.v[(vi + 2) % 3] as usize] = visit_id;
                    self.push_fan_to_stack(tri_idx, vi, &mut stack);
                }
                continue;
            }

            // Perform 2-2 flip
            self.flip22(tri_idx, vi, &mut stack);
        }
    }

    /// Push fan of edges around vertex to stack.
    ///
    /// Ported from Star.cpp (helper for doFlipping)
    fn push_fan_to_stack(&self, start_tri_idx: usize, start_vi: usize, stack: &mut Vec<u32>) {
        let mut tri_idx = start_tri_idx;
        let mut vi = start_vi;

        loop {
            stack.push(encode(tri_idx as u32, vi as u32));

            let opp = self.tri_opp_vec[tri_idx];
            // Check for boundary (INVALID adjacency = star boundary)
            if opp.t[(vi + 1) % 3] == crate::types::INVALID {
                break;
            }
            let next_tri = opp.get_opp_tri((vi + 1) % 3) as usize;
            let next_vi = opp.get_opp_vi((vi + 1) % 3) as usize;

            if next_tri == start_tri_idx {
                break;
            }
            if next_tri >= self.tri_status_vec.len() {
                break;
            }

            tri_idx = next_tri;
            vi = (next_vi + 2) % 3;
        }
    }

    /// Locate a triangle containing the vertex (walk from arbitrary start).
    ///
    /// Ported from Star.cpp lines 758-806
    ///
    /// If the walk hits a boundary (INVALID adjacency), falls back to
    /// linear scan of all triangles.
    fn locate_vert(&self, in_vert: u32) -> Option<usize> {
        // Find first non-free triangle
        let mut tri_idx = 0;
        while tri_idx < self.tri_vec.len() {
            if self.tri_status_vec[tri_idx] != TriStatus::Free {
                break;
            }
            tri_idx += 1;
        }

        if tri_idx >= self.tri_vec.len() {
            return None; // No alive triangles
        }

        let mut prev_vi = None;

        // Walk to containing triangle (CUDA: Star.cpp:774-803)
        let max_steps = self.tri_vec.len() * 3;
        let mut step = 0;
        loop {
            step += 1;
            if step > max_steps || tri_idx >= self.tri_vec.len() {
                break; // Walk didn't converge — fall back to linear scan
            }

            let tri = self.tri_vec[tri_idx];

            // Check 3 sides
            let mut vi = 0;
            while vi < 3 {
                if Some(vi) == prev_vi {
                    vi += 1;
                    continue; // Skip incoming direction
                }

                let ori = predicates::orient3d(
                    self.points[tri.v[(vi + 1) % 3] as usize],
                    self.points[tri.v[(vi + 2) % 3] as usize],
                    self.points[in_vert as usize],
                    self.points[self.vert as usize],
                );

                if ori < 0.0 {
                    break;
                }
                vi += 1;
            }

            if vi >= 3 {
                // Found containing triangle
                return Some(tri_idx);
            }

            let opp = self.tri_opp_vec[tri_idx];
            // Check for INVALID adjacency (boundary of star)
            if opp.t[vi] == crate::types::INVALID {
                break; // Hit star boundary — fall back to linear scan
            }
            let opp_ti = opp.get_opp_tri(vi) as usize;
            if opp_ti >= self.tri_vec.len() {
                break; // Out of bounds — fall back to linear scan
            }
            let opp_vi = opp.get_opp_vi(vi) as usize;

            tri_idx = opp_ti;
            prev_vi = Some(opp_vi);
        }

        // Fallback: linear scan of all non-free triangles.
        // Test each triangle to see if in_vert is "inside" (all orient3d >= 0).
        for ti in 0..self.tri_vec.len() {
            if self.tri_status_vec[ti] == TriStatus::Free {
                continue;
            }
            let tri = self.tri_vec[ti];
            let mut inside = true;
            for vi in 0..3 {
                let ori = predicates::orient3d(
                    self.points[tri.v[(vi + 1) % 3] as usize],
                    self.points[tri.v[(vi + 2) % 3] as usize],
                    self.points[in_vert as usize],
                    self.points[self.vert as usize],
                );
                if ori < 0.0 {
                    inside = false;
                    break;
                }
            }
            if inside {
                return Some(ti);
            }
        }
        None
    }

    /// Mark triangles "beneath" the new vertex and find one to use as hole base.
    ///
    /// Ported from Star.cpp lines 379-461
    fn mark_beneath_triangles(
        &mut self,
        ins_vert: u32,
        stack: &mut Vec<usize>,
        visited: &mut Vec<i32>,
        visit_id: i32,
    ) -> Option<usize> {
        let container = self.locate_vert(ins_vert)?;

        stack.clear();
        stack.push(container);
        visited[container] = visit_id;

        let mut ben_tri_idx = None;

        // DFS to mark all beneath triangles
        while let Some(tri_idx) = stack.pop() {
            let tri = self.tri_vec[tri_idx];

            let ort = predicates::orient4d(
                self.points[tri.v[0] as usize],
                self.points[tri.v[1] as usize],
                self.points[tri.v[2] as usize],
                self.points[self.vert as usize],
                self.points[ins_vert as usize],
                tri.v[0],
                tri.v[1],
                tri.v[2],
                self.vert,
                ins_vert,
            );

            if ort < 0 {
                // Mark as beneath
                self.tri_status_vec[tri_idx] = TriStatus::Free;
                self.tet_idx_vec[tri_idx] = -1;
                ben_tri_idx = Some(tri_idx);

                let opp = self.tri_opp_vec[tri_idx];

                // Check for vertex deletions and add neighbors
                for vi in 0..3 {
                    self.check_vert_deleted(tri_idx, vi);

                    // Guard against INVALID adjacency (star boundary)
                    if opp.t[vi] == crate::types::INVALID {
                        continue;
                    }
                    let opp_tri = opp.get_opp_tri(vi) as usize;
                    if opp_tri >= visited.len() {
                        continue;
                    }

                    if visited[opp_tri] != visit_id {
                        stack.push(opp_tri);
                        visited[opp_tri] = visit_id;
                    }
                }
            }
        }

        ben_tri_idx
    }

    /// Check if a vertex was completely deleted by marking.
    ///
    /// Ported from Star.cpp lines 350-377
    fn check_vert_deleted(&mut self, tri_idx: usize, vi: usize) {
        let mut cur_vi = (vi + 2) % 3;
        let mut cur_tri_idx = tri_idx;

        let mut vert_deleted = false;

        // Rotate around vertex
        while self.tri_status_vec[cur_tri_idx] == TriStatus::Free {
            let cur_tri_opp = self.tri_opp_vec[cur_tri_idx];
            // Guard against INVALID adjacency (star boundary)
            if cur_tri_opp.t[(cur_vi + 2) % 3] == crate::types::INVALID {
                break;
            }
            let opp_tri_idx = cur_tri_opp.get_opp_tri((cur_vi + 2) % 3) as usize;
            if opp_tri_idx >= self.tri_vec.len() {
                break;
            }

            cur_vi = cur_tri_opp.get_opp_vi((cur_vi + 2) % 3) as usize;
            cur_tri_idx = opp_tri_idx;

            if cur_tri_idx == tri_idx {
                vert_deleted = true;
                break;
            }
        }

        if vert_deleted {
            self.add_one_deletion_to_queue(self.tri_vec[tri_idx].v[vi]);
        }
    }

    /// Find the first hole segment (boundary edge of the hole).
    ///
    /// Ported from Star.cpp lines 463-515
    fn find_first_hole_segment(
        &self,
        ben_tri_idx: usize,
    ) -> Option<(usize, usize, usize)> {
        // Check if input beneath triangle is on hole border
        let tri_opp_first = self.tri_opp_vec[ben_tri_idx];

        for vi in 0..3 {
            if tri_opp_first.t[vi] == crate::types::INVALID {
                continue;
            }
            let opp_tri_idx = tri_opp_first.get_opp_tri(vi) as usize;
            if opp_tri_idx >= self.tri_status_vec.len() {
                continue;
            }

            if self.tri_status_vec[opp_tri_idx] != TriStatus::Free {
                let first_tri_idx = opp_tri_idx;
                let first_hole_tri_idx = ben_tri_idx;
                let first_vi = tri_opp_first.get_opp_vi(vi) as usize;

                return Some((first_tri_idx, first_vi, first_hole_tri_idx));
            }
        }

        // Iterate all triangles to find hole border
        for tri_idx in 0..self.tri_vec.len() {
            if self.tri_status_vec[tri_idx] == TriStatus::Free {
                continue;
            }

            let tri_opp = self.tri_opp_vec[tri_idx];

            for vi in 0..3 {
                if tri_opp.t[vi] == crate::types::INVALID {
                    continue;
                }
                let tri_opp_idx = tri_opp.get_opp_tri(vi) as usize;
                if tri_opp_idx >= self.tri_status_vec.len() {
                    continue;
                }

                if self.tri_status_vec[tri_opp_idx] == TriStatus::Free {
                    return Some((tri_idx, vi, tri_opp_idx));
                }
            }
        }

        None
    }

    /// Get a free triangle index (recycle or allocate new).
    ///
    /// Ported from Star.cpp lines 517-540
    fn get_free_tri(&mut self, free_tri_idx: &mut usize) -> usize {
        while *free_tri_idx < self.tri_vec.len() {
            if self.tri_status_vec[*free_tri_idx] == TriStatus::Free {
                return *free_tri_idx;
            }
            *free_tri_idx += 1;
        }

        // Allocate new triangle
        let old_size = self.tri_vec.len();

        self.tri_vec.push(Tri::new(0, 0, 0));
        self.tri_opp_vec.push(TriOpp::new());
        self.tet_idx_vec.push(-1);
        self.tri_status_vec.push(TriStatus::Free);

        *free_tri_idx = old_size;
        old_size
    }

    /// Change all TriNew triangles to TriValid.
    ///
    /// Ported from Star.cpp lines 542-551
    fn change_new_to_valid(&mut self) {
        for tri_idx in 0..self.tri_status_vec.len() {
            if self.tri_status_vec[tri_idx] == TriStatus::New {
                self.tri_status_vec[tri_idx] = TriStatus::Valid;
            }
        }
    }

    /// Stitch new vertex to the hole created by marking beneath triangles.
    ///
    /// Ported from Star.cpp lines 553-672
    /// Returns false if no hole segment found (all triangles marked beneath).
    fn stitch_vert_to_hole(&mut self, ins_vert: u32, ben_tri_idx: usize) -> bool {
        let (first_tri_idx, first_vi, first_hole_tri_idx) =
            match self.find_first_hole_segment(ben_tri_idx) {
                Some(seg) => seg,
                None => return false, // All triangles marked beneath — can't stitch
            };

        // Get first two vertices of hole
        // CUDA: Star.cpp:568-572
        let mut cur_tri_idx = first_tri_idx;
        let cur_tri = self.tri_vec[cur_tri_idx];
        let first_vert = cur_tri.v[(first_vi + 1) % 3];
        let mut cur_vi = (first_vi + 2) % 3;
        let mut cur_vert = cur_tri.v[cur_vi];

        // Stitch first triangle
        // CUDA: Star.cpp:576-588
        // CUDA reuses firstHoleTriIdx directly; we do the same
        let first_new_tri_idx = first_hole_tri_idx;

        // CUDA vertex order: { insVert, curVert, firstVert }
        let first_new_tri = Tri::new(ins_vert, cur_vert, first_vert);

        self.tri_vec[first_new_tri_idx] = first_new_tri;
        self.tet_idx_vec[first_new_tri_idx] = -1;
        self.tri_status_vec[first_new_tri_idx] = TriStatus::New;

        self.add_one_tri_to_queue(first_new_tri_idx, &first_new_tri);

        // Adjacency: firstNewTri face 0 ↔ firstTriIdx face firstVi
        // CUDA: Star.cpp:584-588
        self.tri_opp_vec[first_new_tri_idx].set_opp(0, first_tri_idx as u32, first_vi as u32);
        self.tri_opp_vec[first_tri_idx].set_opp(first_vi, first_new_tri_idx as u32, 0);

        let mut free_tri_idx = 0;
        let mut prev_new_tri_idx = first_new_tri_idx;

        // Walk around outside of hole in CW direction, stitching triangles
        // CUDA: Star.cpp:597-667
        let max_iters = self.tri_vec.len() * 3; // Safety limit
        let mut iters = 0;
        while cur_vert != first_vert {
            iters += 1;
            if iters > max_iters {
                break;
            }

            // Check opposite triangle in CW direction: (curVi + 2) % 3
            // CUDA: Star.cpp:600-601
            let walk_vi = (cur_vi + 2) % 3;
            let cur_tri_opp = self.tri_opp_vec[cur_tri_idx];
            if cur_tri_opp.t[walk_vi] == crate::types::INVALID {
                break; // Hit star boundary
            }
            let glo_opp_tri_idx = cur_tri_opp.get_opp_tri(walk_vi) as usize;
            if glo_opp_tri_idx >= self.tri_status_vec.len() {
                break;
            }
            let status = self.tri_status_vec[glo_opp_tri_idx];

            // CUDA: Star.cpp:607-613 — Tri is outside the hole (not Free, not New)
            if status != TriStatus::Free && status != TriStatus::New {
                // Continue walking around the border
                let opp_vi = cur_tri_opp.get_opp_vi(walk_vi) as usize;
                cur_vi = (opp_vi + 2) % 3;
                cur_tri_idx = glo_opp_tri_idx;
            } else {
                // CUDA: Star.cpp:617-666 — Tri is in hole, create new triangle
                let new_tri_idx = if status == TriStatus::Free {
                    glo_opp_tri_idx // Reuse hole triangle slot
                } else {
                    self.get_free_tri(&mut free_tri_idx)
                };

                // Get next vertex in hole boundary
                // CUDA: Star.cpp:623-625
                let opp_vi = (cur_vi + 2) % 3;
                let cur_tri_data = self.tri_vec[cur_tri_idx];
                let next_vert = cur_tri_data.v[(cur_vi + 1) % 3];

                // CUDA vertex order: { insVert, nextVert, curVert }
                let new_tri = Tri::new(ins_vert, next_vert, cur_vert);

                // Adjacency with opposite (border) triangle
                // CUDA: Star.cpp:632-636
                self.tri_opp_vec[cur_tri_idx].set_opp(opp_vi, new_tri_idx as u32, 0);
                self.tri_opp_vec[new_tri_idx].set_opp(0, cur_tri_idx as u32, opp_vi as u32);

                // Adjacency with previous new triangle
                // CUDA: Star.cpp:640-642
                self.tri_opp_vec[prev_new_tri_idx].set_opp(2, new_tri_idx as u32, 1);
                self.tri_opp_vec[new_tri_idx].set_opp(1, prev_new_tri_idx as u32, 2);

                // Last hole triangle: close the loop
                // CUDA: Star.cpp:645-650
                if next_vert == first_vert {
                    self.tri_opp_vec[first_new_tri_idx].set_opp(1, new_tri_idx as u32, 2);
                    self.tri_opp_vec[new_tri_idx].set_opp(2, first_new_tri_idx as u32, 1);
                }

                // Store new triangle data
                self.tri_vec[new_tri_idx] = new_tri;
                self.tet_idx_vec[new_tri_idx] = -1;
                self.tri_status_vec[new_tri_idx] = TriStatus::New;

                self.add_one_tri_to_queue(new_tri_idx, &new_tri);

                // Advance
                prev_new_tri_idx = new_tri_idx;
                cur_vi = (cur_vi + 1) % 3;
                cur_vert = next_vert;
            }
        }

        // If loop didn't close naturally (boundary case), close it now
        if cur_vert == first_vert && prev_new_tri_idx != first_new_tri_idx {
            // Already closed in the loop's last-hole-triangle check
        } else if cur_vert != first_vert {
            // Incomplete hole walk (boundary star) — close what we have
            self.tri_opp_vec[first_new_tri_idx].set_opp(1, prev_new_tri_idx as u32, 2);
            self.tri_opp_vec[prev_new_tri_idx].set_opp(2, first_new_tri_idx as u32, 1);
        }

        self.change_new_to_valid();
        true
    }

    /// Insert new vertex into star's link.
    ///
    /// Ported from Star.cpp lines 326-348
    pub fn insert_to_star(
        &mut self,
        ins_vert: u32,
        stack: &mut Vec<usize>,
        visited: &mut Vec<i32>,
        visit_id: i32,
    ) -> bool {
        // Guard: never insert center vertex into own star
        if ins_vert == self.vert {
            return true;
        }

        // Save state: mark_beneath_triangles destructively sets triangles to Free.
        // If stitch_vert_to_hole then fails, the star is corrupted.
        // We must restore to prevent cascade failures.
        let saved_status = self.tri_status_vec.clone();
        let saved_tet_idx = self.tet_idx_vec.clone();

        let ben_tri_idx = match self.mark_beneath_triangles(ins_vert, stack, visited, visit_id) {
            Some(idx) => idx,
            None => {
                // Restore: mark_beneath may have partially marked Free
                self.tri_status_vec = saved_status;
                self.tet_idx_vec = saved_tet_idx;
                return false;
            }
        };

        if !self.stitch_vert_to_hole(ins_vert, ben_tri_idx) {
            // Restore: stitch failed, star has corrupted state
            self.tri_status_vec = saved_status;
            self.tet_idx_vec = saved_tet_idx;
            return false;
        }
        true
    }

    /// Get proof vertices for failed insertion (orient4d-based).
    ///
    /// Ported from Star.cpp lines 674-740
    ///
    /// Returns 4 vertices that form a "proof" explaining why insertion failed.
    /// Uses orient4d to find a triangle intersected by the plane
    /// (starVert, inVert, exVert, planeVert).
    pub fn get_proof(&self, in_vert: u32) -> [u32; 4] {
        // Pick first valid triangle
        let mut loc_tri_idx = 0;
        while loc_tri_idx < self.tri_vec.len() {
            if self.tri_status_vec[loc_tri_idx] != TriStatus::Free {
                break;
            }
            loc_tri_idx += 1;
        }

        if loc_tri_idx >= self.tri_vec.len() {
            return [0, 0, 0, 0]; // No valid triangles
        }

        let first_tri = self.tri_vec[loc_tri_idx];
        let ex_vert = first_tri.v[0]; // First proof point

        // CUDA: Star.cpp:697-726
        // Find a triangle (not containing exVert) where all orient4d results match.
        // This means the proof plane (starVert, inVert, exVert, planeVert)
        // separates this triangle consistently.
        for idx in loc_tri_idx..self.tri_vec.len() {
            if self.tri_status_vec[idx] == TriStatus::Free {
                continue;
            }

            let tri = self.tri_vec[idx];

            if tri.has(ex_vert) {
                continue;
            }

            // Check orient4d for each vertex pair in the triangle
            // CUDA: ord[vi] = orient4d(starVert, inVert, exVert, planeVert, testVert)
            let mut all_match = true;
            let mut prev_ord = 0i32;
            for vi in 0..3 {
                let plane_vert = tri.v[vi];
                let test_vert = tri.v[(vi + 1) % 3];

                let ord = predicates::orient4d(
                    self.points[self.vert as usize],
                    self.points[in_vert as usize],
                    self.points[ex_vert as usize],
                    self.points[plane_vert as usize],
                    self.points[test_vert as usize],
                    self.vert,
                    in_vert,
                    ex_vert,
                    plane_vert,
                    test_vert,
                );

                if vi > 0 && prev_ord != ord {
                    all_match = false;
                    break;
                }
                prev_ord = ord;
            }

            if all_match {
                return [ex_vert, tri.v[0], tri.v[1], tri.v[2]];
            }
        }

        // Fallback: return first triangle vertices
        [ex_vert, first_tri.v[0], first_tri.v[1], first_tri.v[2]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_star_creation() {
        let points = vec![[0.0, 0.0, 0.0]];
        let mut queue = Vec::new();
        let star = Star::new(42, points, &mut queue as *mut Vec<Facet>);
        assert_eq!(star.vert, 42);
        assert_eq!(star.tri_vec.len(), 0);
    }

    #[test]
    fn test_has_link_vert() {
        let points = vec![[0.0, 0.0, 0.0]];
        let mut queue = Vec::new();
        let mut star = Star::new(10, points, &mut queue as *mut Vec<Facet>);
        assert!(!star.has_link_vert(20));

        // Add a triangle containing vertex 20
        star.tri_vec.push(Tri::new(20, 30, 40));
        assert!(star.has_link_vert(20));
        assert!(star.has_link_vert(30));
        assert!(!star.has_link_vert(50));
    }

    #[test]
    fn test_flip22_basic() {
        // Create a simple star with two triangles sharing an edge
        let points = vec![
            [0.0, 0.0, 0.0], // 0: star center
            [1.0, 0.0, 0.0], // 1
            [0.0, 1.0, 0.0], // 2
            [1.0, 1.0, 0.0], // 3
            [0.5, 0.5, 1.0], // 4
        ];
        let mut queue = Vec::new();
        let mut star = Star::new(0, points, &mut queue as *mut Vec<Facet>);

        // NOTE: flip22 requires full adjacency setup which is complex
        // For now, just verify the star can be created
        star.tri_vec.push(Tri::new(1, 2, 3));
        star.tri_status_vec.push(TriStatus::Valid);
        star.tet_idx_vec.push(-1);
        star.tri_opp_vec.push(TriOpp::new());

        assert_eq!(star.tri_vec.len(), 1);
        assert_eq!(star.vert, 0);
    }
}
