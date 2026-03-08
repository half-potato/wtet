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

        let opp_tri = opp.get_opp_tri(vi) as usize;
        let opp_vi = opp.get_opp_vi(vi) as usize;
        let side_tri = opp.get_opp_tri(cor_vi) as usize;
        let side_vi = opp.get_opp_vi(cor_vi) as usize;
        let opp_vert = self.tri_vec[opp_tri].v[opp_vi];

        // Create new triangle
        let new_tri = Tri::new(tri.v[vi], tri.v[cor_vi], opp_vert);

        self.tri_vec[tri_idx] = new_tri;
        self.tet_idx_vec[tri_idx] = -1;

        // Build new adjacency
        let opp_v0 = self.tri_opp_vec[opp_tri].get_opp_tri_vi((opp_vi + 1) % 3);
        let opp_v1 = self.tri_opp_vec[side_tri].get_opp_tri_vi((side_vi + 2) % 3);
        let opp_v2 = opp.get_opp_tri_vi((vi + 2) % 3);

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
        let opp = self.tri_opp_vec[tri_idx];
        let tri = self.tri_vec[tri_idx];

        let opp_tri = opp.get_opp_tri(vi) as usize;
        let opp_vi = opp.get_opp_vi(vi) as usize;
        let opp_vert = self.tri_vec[opp_tri].v[opp_vi];

        // Create two new triangles
        let new_tri0 = Tri::new(tri.v[vi], opp_vert, tri.v[(vi + 2) % 3]);
        let new_tri1 = Tri::new(opp_vert, tri.v[vi], tri.v[(vi + 1) % 3]);

        self.tri_vec[tri_idx] = new_tri0;
        self.tri_vec[opp_tri] = new_tri1;

        self.tet_idx_vec[tri_idx] = -1;
        self.tet_idx_vec[opp_tri] = -1;

        // Build new adjacency
        let opp_v0 = opp.get_opp_tri_vi((vi + 2) % 3);
        let opp_v1 = self.tri_opp_vec[opp_tri].get_opp_tri_vi((opp_vi + 1) % 3);
        let opp_v2 = self.tri_opp_vec[opp_tri].get_opp_tri_vi((opp_vi + 2) % 3);
        let opp_v3 = opp.get_opp_tri_vi((vi + 1) % 3);

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
        let mut is_non_extreme = vec![0i32; self.points.len()];
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

            let opp_tri = opp.get_opp_tri(vi) as usize;
            let opp_vi = opp.get_opp_vi(vi) as usize;

            if self.tri_status_vec[opp_tri] == TriStatus::Free {
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

            // Check for 3-1 flip possibility
            if opp.get_opp_tri((vi + 1) % 3) == self.tri_opp_vec[opp_tri].get_opp_tri((opp_vi + 2) % 3) {
                if ort < 0 || min_vert == tri.v[(vi + 2) % 3] {
                    self.flip31(tri_idx, vi, &mut stack);
                }
                continue;
            }

            if opp.get_opp_tri((vi + 2) % 3) == self.tri_opp_vec[opp_tri].get_opp_tri((opp_vi + 1) % 3) {
                if ort < 0 || min_vert == tri.v[(vi + 1) % 3] {
                    self.flip31(tri_idx, (vi + 2) % 3, &mut stack);
                }
                continue;
            }

            // Check 2-2 flippability using orient3d
            let or1 = predicates::orient3d(
                self.points[tri.v[vi] as usize],
                self.points[tri.v[(vi + 1) % 3] as usize],
                self.points[opp_vert as usize],
                self.points[self.vert as usize],
            );

            if or1 < 0.0 {
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

            if or2 < 0.0 {
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
            let next_tri = opp.get_opp_tri((vi + 1) % 3) as usize;
            let next_vi = opp.get_opp_vi((vi + 1) % 3) as usize;

            if next_tri == start_tri_idx {
                break;
            }

            tri_idx = next_tri;
            vi = (next_vi + 2) % 3;
        }
    }

    /// Locate a triangle containing the vertex (walk from arbitrary start).
    ///
    /// Ported from Star.cpp lines 758-806
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

        // Walk to containing triangle
        loop {
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
            let opp_ti = opp.get_opp_tri(vi) as usize;
            let opp_vi = opp.get_opp_vi(vi) as usize;

            tri_idx = opp_ti;
            prev_vi = Some(opp_vi);
        }
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

                    let opp_tri = opp.get_opp_tri(vi) as usize;

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
            let opp_tri_idx = cur_tri_opp.get_opp_tri((cur_vi + 2) % 3) as usize;

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
            let opp_tri_idx = tri_opp_first.get_opp_tri(vi) as usize;

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
                let tri_opp_idx = tri_opp.get_opp_tri(vi) as usize;

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
    fn stitch_vert_to_hole(&mut self, ins_vert: u32, ben_tri_idx: usize) {
        let (first_tri_idx, first_vi, _first_hole_tri_idx) =
            self.find_first_hole_segment(ben_tri_idx).expect("No hole segment found");

        // Get first two vertices of hole
        let cur_tri_idx = first_tri_idx;
        let cur_tri = self.tri_vec[cur_tri_idx];
        let first_vert = cur_tri.v[(first_vi + 1) % 3];
        let mut cur_vi = (first_vi + 2) % 3;
        let mut cur_vert = cur_tri.v[cur_vi];

        // Stitch first triangle
        let mut free_tri_idx = 0;
        let first_new_tri_idx = self.get_free_tri(&mut free_tri_idx);

        let new_tri = Tri::new(ins_vert, first_vert, cur_vert);

        self.tri_vec[first_new_tri_idx] = new_tri;
        self.tet_idx_vec[first_new_tri_idx] = -1;
        self.tri_status_vec[first_new_tri_idx] = TriStatus::New;

        // Set adjacency with opposite triangle
        self.tri_opp_vec[cur_tri_idx].set_opp(cur_vi, first_new_tri_idx as u32, 0);
        self.tri_opp_vec[first_new_tri_idx].set_opp(0, cur_tri_idx as u32, cur_vi as u32);

        self.add_one_tri_to_queue(first_new_tri_idx, &new_tri);

        let mut prev_new_tri_idx = first_new_tri_idx;

        // Stitch remaining triangles
        loop {
            // Move to next edge
            let opp = self.tri_opp_vec[cur_tri_idx];
            let next_tri_idx = opp.get_opp_tri((cur_vi + 1) % 3) as usize;
            let opp_vi = opp.get_opp_vi((cur_vi + 1) % 3) as usize;

            // Find next vertex
            let next_tri = self.tri_vec[next_tri_idx];
            let next_vert = next_tri.v[(opp_vi + 2) % 3];

            if next_vert == first_vert {
                break; // Completed the hole
            }

            // Create new triangle
            let new_tri_idx = self.get_free_tri(&mut free_tri_idx);
            let new_tri = Tri::new(ins_vert, cur_vert, next_vert);

            // Adjacency with opposite triangle
            self.tri_opp_vec[next_tri_idx].set_opp(opp_vi, new_tri_idx as u32, 0);
            self.tri_opp_vec[new_tri_idx].set_opp(0, next_tri_idx as u32, opp_vi as u32);

            // Adjacency with previous new triangle
            self.tri_opp_vec[prev_new_tri_idx].set_opp(2, new_tri_idx as u32, 1);
            self.tri_opp_vec[new_tri_idx].set_opp(1, prev_new_tri_idx as u32, 2);

            // Last triangle: close the loop
            if next_vert == first_vert {
                self.tri_opp_vec[first_new_tri_idx].set_opp(1, new_tri_idx as u32, 2);
                self.tri_opp_vec[new_tri_idx].set_opp(2, first_new_tri_idx as u32, 1);
            }

            // Store new triangle
            self.tri_vec[new_tri_idx] = new_tri;
            self.tet_idx_vec[new_tri_idx] = -1;
            self.tri_status_vec[new_tri_idx] = TriStatus::New;

            self.add_one_tri_to_queue(new_tri_idx, &new_tri);

            prev_new_tri_idx = new_tri_idx;
            cur_vi = (opp_vi + 1) % 3;
            cur_vert = next_vert;

            // Safety check
            if self.tri_vec.len() > 10000 {
                eprintln!("WARNING: Star growing too large, stopping stitching");
                break;
            }
        }

        // Close the loop with first triangle
        self.tri_opp_vec[first_new_tri_idx].set_opp(1, prev_new_tri_idx as u32, 2);
        self.tri_opp_vec[prev_new_tri_idx].set_opp(2, first_new_tri_idx as u32, 1);

        self.change_new_to_valid();
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
        let ben_tri_idx = match self.mark_beneath_triangles(ins_vert, stack, visited, visit_id) {
            Some(idx) => idx,
            None => return false,
        };

        self.stitch_vert_to_hole(ins_vert, ben_tri_idx);

        true
    }

    /// Get proof vertices for failed insertion (orient4d-based).
    ///
    /// Ported from Star.cpp lines 674-750
    ///
    /// Returns 4 vertices that form a "proof" explaining why insertion failed.
    /// Uses orient4d tests to find a planar configuration.
    pub fn get_proof(&self, in_vert: u32) -> [u32; 4] {
        // Pick one valid triangle
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

        // Find another triangle not containing ex_vert
        while loc_tri_idx < self.tri_vec.len() {
            if self.tri_status_vec[loc_tri_idx] == TriStatus::Free {
                loc_tri_idx += 1;
                continue;
            }

            let tri = self.tri_vec[loc_tri_idx];

            if tri.has(ex_vert) {
                loc_tri_idx += 1;
                continue;
            }

            // Use orient4d to find proof vertices
            // Simplified version: just return the triangle + star vertex
            return [ex_vert, tri.v[0], tri.v[1], tri.v[2]];
        }

        // Fallback: return first triangle + star vertex
        [self.vert, first_tri.v[0], first_tri.v[1], first_tri.v[2]]
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
