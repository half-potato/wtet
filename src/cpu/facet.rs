//! Data structures for star splaying: triangles, adjacency, and work queue items.
//!
//! Ported from gDel3D/GDelFlipping/src/gDel3D/CPU/CPUDecl.h (lines 47-169)

/// Link triangle status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TriStatus {
    Free = 0,
    Valid = 1,
    New = 2,
}

/// Link triangle - 3 vertices forming a face of the star.
///
/// Ported from CPUDecl.h lines 62-113
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tri {
    pub v: [u32; 3],
}

impl Tri {
    pub fn new(v0: u32, v1: u32, v2: u32) -> Self {
        Self { v: [v0, v1, v2] }
    }

    /// Check if triangle contains vertex.
    #[inline]
    pub fn has(&self, vertex: u32) -> bool {
        self.v[0] == vertex || self.v[1] == vertex || self.v[2] == vertex
    }

    /// Check if triangle contains all three vertices (in any order).
    #[inline]
    pub fn has_all(&self, v0: u32, v1: u32, v2: u32) -> bool {
        self.has(v0) && self.has(v1) && self.has(v2)
    }

    /// Get local index of vertex in triangle (0, 1, or 2).
    /// Panics if vertex is not in triangle.
    #[inline]
    pub fn index_of(&self, vertex: u32) -> usize {
        if self.v[0] == vertex {
            0
        } else if self.v[1] == vertex {
            1
        } else if self.v[2] == vertex {
            2
        } else {
            panic!("Vertex {} not in triangle {:?}", vertex, self.v)
        }
    }

    /// Get vertex at local index.
    #[inline]
    pub fn at(&self, local_idx: usize) -> u32 {
        self.v[local_idx]
    }
}

impl PartialOrd for Tri {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Tri {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.v.cmp(&other.v)
    }
}

/// Adjacency between link triangles (5-bit encoding like TetOpp).
///
/// Ported from CPUDecl.h lines 116-169
///
/// **Encoding (same as TetOpp):**
/// - Bits 0-1: vi (vertex index in neighbor triangle)
/// - Bit 2: (unused/internal flag)
/// - Bit 3: special flag
/// - Bit 4: sphere_fail flag
/// - Bits 5-31: triangle index
#[derive(Debug, Clone, Copy)]
pub struct TriOpp {
    pub t: [u32; 3],
}

impl TriOpp {
    pub fn new() -> Self {
        Self {
            t: [u32::MAX; 3], // Initialize to INVALID
        }
    }

    /// Set adjacency: this triangle's face `vi` → neighbor triangle `tri_idx` at face `opp_vi`.
    ///
    /// CRITICAL: Uses 5-bit encoding to match TetOpp (see CommonTypes.h:120-122)
    #[inline]
    pub fn set_opp(&mut self, vi: usize, tri_idx: u32, opp_vi: u32) {
        debug_assert!(vi < 3);
        debug_assert!(opp_vi < 3);
        self.t[vi] = (tri_idx << 5) | (opp_vi & 3);
    }

    /// Get neighbor triangle index.
    #[inline]
    pub fn get_opp_tri(&self, vi: usize) -> u32 {
        debug_assert!(vi < 3);
        self.t[vi] >> 5
    }

    /// Get face index in neighbor triangle.
    #[inline]
    pub fn get_opp_vi(&self, vi: usize) -> u32 {
        debug_assert!(vi < 3);
        self.t[vi] & 3
    }

    /// Set only the triangle index (preserve vi and flags).
    #[inline]
    pub fn set_opp_tri(&mut self, vi: usize, tri_idx: u32) {
        debug_assert!(vi < 3);
        self.t[vi] = (self.t[vi] & 0x1F) | (tri_idx << 5);
    }

    /// Set only the vi (preserve triangle index and flags).
    #[inline]
    pub fn set_opp_vi(&mut self, vi: usize, opp_vi: u32) {
        debug_assert!(vi < 3);
        self.t[vi] = (self.t[vi] & !0x3) | (opp_vi & 3);
    }

    /// Set sphere_fail flag (bit 4).
    #[inline]
    pub fn set_opp_sphere_fail(&mut self, vi: usize) {
        debug_assert!(vi < 3);
        self.t[vi] |= 1 << 4;
    }

    /// Check sphere_fail flag (bit 4).
    #[inline]
    pub fn is_opp_sphere_fail(&self, vi: usize) -> bool {
        debug_assert!(vi < 3);
        (self.t[vi] >> 4) & 1 == 1
    }

    /// Set special flag (bit 3).
    #[inline]
    pub fn set_opp_special(&mut self, vi: usize, state: bool) {
        debug_assert!(vi < 3);
        if state {
            self.t[vi] |= 1 << 3;
        } else {
            self.t[vi] &= !(1 << 3);
        }
    }

    /// Check special flag (bit 3).
    #[inline]
    pub fn is_opp_special(&self, vi: usize) -> bool {
        debug_assert!(vi < 3);
        (self.t[vi] >> 3) & 1 == 1
    }

    /// Get packed tri_idx + vi (mask out flags).
    #[inline]
    pub fn get_opp_tri_vi(&self, vi: usize) -> u32 {
        debug_assert!(vi < 3);
        self.t[vi] & !0x1C // Clear bits 2-4 (flags)
    }
}

impl Default for TriOpp {
    fn default() -> Self {
        Self::new()
    }
}

/// Work queue item for processing star inconsistencies.
///
/// Ported from CPUDecl.h lines 47-54
#[derive(Debug, Clone)]
pub struct Facet {
    /// Source vertex (star being processed)
    pub from: u32,
    /// Index in source star's link
    pub from_idx: u32,
    /// Target vertex to insert into
    pub to: u32,
    /// First vertex of problematic face
    pub v0: u32,
    /// Second vertex of problematic face
    pub v1: u32,
}

impl Facet {
    pub fn new(from: u32, from_idx: u32, to: u32, v0: u32, v1: u32) -> Self {
        Self {
            from,
            from_idx,
            to,
            v0,
            v1,
        }
    }
}

/// Helper functions for encoding/decoding (non-member functions in CUDA).

/// Encode triangle index + local vertex index into single u32.
/// NOTE: Uses 2-bit encoding for internal star use (not 5-bit like TriOpp).
/// vi can be 0-3 (for tet vertices during extraction)
#[inline]
pub fn encode(tri_idx: u32, vi: u32) -> u32 {
    debug_assert!(vi < 4); // Allow 0-3 for tet vertices
    (tri_idx << 2) | (vi & 3)
}

/// Decode triangle index + local vertex index from single u32.
#[inline]
pub fn decode(code: u32) -> (u32, u32) {
    (code >> 2, code & 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tri_has() {
        let tri = Tri::new(10, 20, 30);
        assert!(tri.has(10));
        assert!(tri.has(20));
        assert!(tri.has(30));
        assert!(!tri.has(40));
    }

    #[test]
    fn test_tri_has_all() {
        let tri = Tri::new(10, 20, 30);
        assert!(tri.has_all(10, 20, 30));
        assert!(tri.has_all(30, 10, 20)); // Order doesn't matter
        assert!(!tri.has_all(10, 20, 40));
    }

    #[test]
    fn test_tri_index_of() {
        let tri = Tri::new(10, 20, 30);
        assert_eq!(tri.index_of(10), 0);
        assert_eq!(tri.index_of(20), 1);
        assert_eq!(tri.index_of(30), 2);
    }

    #[test]
    #[should_panic]
    fn test_tri_index_of_not_found() {
        let tri = Tri::new(10, 20, 30);
        let _ = tri.index_of(40);
    }

    #[test]
    fn test_tri_ordering() {
        let tri_a = Tri::new(10, 20, 30);
        let tri_b = Tri::new(10, 20, 40);
        let tri_c = Tri::new(10, 30, 20);

        assert!(tri_a < tri_b);
        assert!(tri_a < tri_c);
    }

    #[test]
    fn test_triopp_encode_decode() {
        let mut opp = TriOpp::new();

        // Set face 0 → triangle 42, local face 1
        opp.set_opp(0, 42, 1);

        assert_eq!(opp.get_opp_tri(0), 42);
        assert_eq!(opp.get_opp_vi(0), 1);

        // Verify 5-bit encoding
        assert_eq!(opp.t[0], (42 << 5) | 1);
    }

    #[test]
    fn test_triopp_set_tri_preserves_vi() {
        let mut opp = TriOpp::new();
        opp.set_opp(0, 10, 2);

        // Change only triangle index
        opp.set_opp_tri(0, 99);

        assert_eq!(opp.get_opp_tri(0), 99);
        assert_eq!(opp.get_opp_vi(0), 2); // Should be preserved
    }

    #[test]
    fn test_triopp_set_vi_preserves_tri() {
        let mut opp = TriOpp::new();
        opp.set_opp(0, 42, 0);

        // Change only vi
        opp.set_opp_vi(0, 2);

        assert_eq!(opp.get_opp_tri(0), 42); // Should be preserved
        assert_eq!(opp.get_opp_vi(0), 2);
    }

    #[test]
    fn test_triopp_sphere_fail_flag() {
        let mut opp = TriOpp::new();
        opp.set_opp(1, 10, 2);

        assert!(!opp.is_opp_sphere_fail(1));

        opp.set_opp_sphere_fail(1);
        assert!(opp.is_opp_sphere_fail(1));

        // Should preserve tri_idx and vi
        assert_eq!(opp.get_opp_tri(1), 10);
        assert_eq!(opp.get_opp_vi(1), 2);
    }

    #[test]
    fn test_triopp_special_flag() {
        let mut opp = TriOpp::new();
        opp.set_opp(2, 20, 1);

        assert!(!opp.is_opp_special(2));

        opp.set_opp_special(2, true);
        assert!(opp.is_opp_special(2));

        opp.set_opp_special(2, false);
        assert!(!opp.is_opp_special(2));
    }

    #[test]
    fn test_encode_decode() {
        let code = encode(123, 2);
        let (tri_idx, vi) = decode(code);

        assert_eq!(tri_idx, 123);
        assert_eq!(vi, 2);
    }

    #[test]
    fn test_facet_creation() {
        let facet = Facet::new(10, 5, 20, 100, 200);

        assert_eq!(facet.from, 10);
        assert_eq!(facet.from_idx, 5);
        assert_eq!(facet.to, 20);
        assert_eq!(facet.v0, 100);
        assert_eq!(facet.v1, 200);
    }
}
