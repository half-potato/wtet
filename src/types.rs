use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Mean vertex degree for block-based allocation.
/// Each vertex gets a block of this many tet slots in free_arr.
/// From original gDel3D KerCommon.h line 56
pub const MEAN_VERTEX_DEGREE: u32 = 8;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Which point to insert when a tet "wins" a vote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertionRule {
    /// Nearest to circumcenter (best quality).
    Circumcenter,
    /// Nearest to centroid (cheaper).
    Centroid,
}

impl Default for InsertionRule {
    fn default() -> Self {
        Self::Circumcenter
    }
}

/// Top-level configuration for the Delaunay algorithm.
#[derive(Debug, Clone)]
pub struct GDelConfig {
    pub insertion_rule: InsertionRule,
    pub enable_flipping: bool,
    pub enable_sorting: bool,
    /// Enable CPU Phase 2 star splaying for guaranteed Delaunay.
    pub enable_splaying: bool,
    /// Maximum insertion+flip iterations before giving up.
    pub max_insert_iterations: u32,
    /// Maximum flip iterations per insertion round.
    pub max_flip_iterations: u32,
}

impl Default for GDelConfig {
    fn default() -> Self {
        Self {
            insertion_rule: InsertionRule::Circumcenter,
            enable_flipping: true,
            enable_sorting: true,
            enable_splaying: true,
            max_insert_iterations: 100,
            max_flip_iterations: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Output of the Delaunay triangulation.
#[derive(Debug, Clone)]
pub struct DelaunayResult {
    /// Tetrahedra: each entry is 4 vertex indices into the original point array.
    pub tets: Vec<[u32; 4]>,
    /// Adjacency: `adjacency[i][f]` is the index of the tet opposite face `f`
    /// of tet `i`, with the face index packed in the low 2 bits.
    /// `u32::MAX` means boundary (no neighbour).
    pub adjacency: Vec<[u32; 4]>,
    /// Vertices that failed to satisfy Delaunay after all iterations.
    /// Should be empty for a correct triangulation.
    pub failed_verts: Vec<u32>,
}

// ---------------------------------------------------------------------------
// GPU-side types (Pod for buffer upload/download)
// ---------------------------------------------------------------------------

/// Tet vertex indices — maps to `vec4<u32>` in WGSL.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GpuTet {
    pub v: [u32; 4],
}

/// Opposite tet+face encoding — maps to `vec4<u32>` in WGSL.
/// For face `f` of tet `t`: `opp[f]` encodes `(neighbour_tet_idx << 2) | neighbour_face`.
/// `u32::MAX` (0xFFFFFFFF) = no neighbour.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GpuTetOpp {
    pub opp: [u32; 4],
}

/// Per-tet status flags.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GpuTetInfo {
    pub flags: u32,
}

// Tet status flag bits
pub const TET_ALIVE: u32 = 1 << 0;
pub const TET_CHANGED: u32 = 1 << 1;
pub const TET_CHECKED: u32 = 1 << 2;
pub const TET_LOCKED: u32 = 1 << 3;

/// Point — padded to vec4 for GPU alignment. `w` is unused (set to 1.0).
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GpuPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl GpuPoint {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z, w: 1.0 }
    }

    /// The "point at infinity" — used by super-tetrahedron.
    pub fn infinity() -> Self {
        Self {
            x: f32::INFINITY,
            y: f32::INFINITY,
            z: f32::INFINITY,
            w: 0.0,
        }
    }
}

/// Atomic counters used by various kernels.
/// Laid out as a `array<atomic<u32>, 8>` on GPU.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GpuCounters {
    /// Number of free slots in the free stack.
    pub free_count: u32,
    /// Number of currently active (alive) tets.
    pub active_count: u32,
    /// Number of inserted points this iteration.
    pub inserted_count: u32,
    /// Number of failed vertices.
    pub failed_count: u32,
    /// Scratch counters.
    pub scratch: [u32; 4],
}

// Counter indices (for shader access)
pub const COUNTER_FREE: u32 = 0;
pub const COUNTER_ACTIVE: u32 = 1;
pub const COUNTER_INSERTED: u32 = 2;
pub const COUNTER_FAILED: u32 = 3;

/// Encodes a tet index + face index into a single u32.
pub fn encode_opp(tet_idx: u32, face: u32) -> u32 {
    debug_assert!(face < 4);
    (tet_idx << 2) | (face & 3)
}

/// Decodes a tet index + face index from a packed u32.
pub fn decode_opp(packed: u32) -> (u32, u32) {
    (packed >> 2, packed & 3)
}

/// Sentinel value meaning "no neighbour" / "invalid".
pub const INVALID: u32 = u32::MAX;
