pub mod buffers;
pub mod pipelines;
pub mod dispatch;

use crate::types::{*, MEAN_VERTEX_DEGREE};
use buffers::GpuBuffers;
use pipelines::Pipelines;

/// Holds all GPU state: buffers, pipelines, and configuration.
pub struct GpuState {
    pub buffers: GpuBuffers,
    pub pipelines: Pipelines,
    pub num_points: u32,
    pub max_tets: u32,
    /// Current number of tets in use (including dead tets)
    pub current_tet_num: u32,
    /// Points that still need to be inserted (indices into the point array).
    pub uninserted: Vec<u32>,
}

impl GpuState {
    pub async fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        points: &[[f32; 3]],
        _config: &GDelConfig,
    ) -> Self {
        eprintln!("\n[GPU STATE] Initializing GpuState...");
        let num_points = points.len() as u32;
        eprintln!("[GPU STATE] Input: {} points", num_points);
        // For block-based allocation, need enough tets to fill all vertex blocks
        // Each vertex (including 4 super-tet vertices) gets MEAN_VERTEX_DEGREE slots
        // Plus MEAN_VERTEX_DEGREE slots for the infinity block (used during flips)
        let min_tets_for_blocks = (num_points + 4) * MEAN_VERTEX_DEGREE + MEAN_VERTEX_DEGREE;
        // Also consider typical Delaunay tet count (~6.5x points) + overhead for retries
        // Allocate generously: 30x points to handle worst-case insertions + flips + retries
        let typical_tets = num_points * 30;
        let max_tets = min_tets_for_blocks.max(typical_tets).max(64);

        // Build point buffer: N real points + 4 super-tet vertices
        let mut gpu_points: Vec<GpuPoint> = points
            .iter()
            .map(|p| GpuPoint::new(p[0], p[1], p[2]))
            .collect();

        // Super-tetrahedron vertices: large enough to contain [0,1]³
        // We place them far outside so all real points are inside.
        let big = 100.0_f32;
        gpu_points.push(GpuPoint::new(-big, -big, -big));       // n+0
        gpu_points.push(GpuPoint::new(4.0 * big, -big, -big));  // n+1
        gpu_points.push(GpuPoint::new(-big, 4.0 * big, -big));  // n+2
        gpu_points.push(GpuPoint::new(-big, -big, 4.0 * big));  // n+3

        eprintln!("[GPU STATE] Creating buffers...");
        let buffers = GpuBuffers::new(device, &gpu_points, num_points, max_tets);
        eprintln!("[GPU STATE] ✓ Buffers created");

        eprintln!("[GPU STATE] Creating pipelines...");
        let pipelines = Pipelines::new(device, &buffers, num_points, max_tets);
        eprintln!("[GPU STATE] ✓ Pipelines created");

        // All real points start as uninserted
        let uninserted: Vec<u32> = (0..num_points).collect();

        eprintln!("[GPU STATE] ✓ GpuState initialization complete\n");

        Self {
            buffers,
            pipelines,
            num_points,
            max_tets,
            current_tet_num: 1, // Start with 1 (super-tet created by init kernel)
            uninserted,
        }
    }

    /// Read back tets, adjacency, and failed verts from GPU to CPU.
    pub async fn readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> DelaunayResult {
        let num_points = self.num_points;

        let counters = self.buffers.read_counters(device, queue).await;

        // Read full tet, adjacency, and info buffers (tets are scattered, not contiguous)
        let max = self.max_tets as usize;
        let tets_raw = self.buffers.read_tets(device, queue, max).await;
        let opp_raw = self.buffers.read_opp(device, queue, max).await;
        let info_raw: Vec<GpuTetInfo> = self
            .buffers
            .read_buffer_as(device, queue, &self.buffers.tet_info, max)
            .await;

        // Filter: keep only alive tets that don't reference super-tet vertices
        let mut tets = Vec::new();
        let mut adjacency = Vec::new();

        for i in 0..max {
            if (info_raw[i].flags & TET_ALIVE) == 0 {
                continue;
            }
            let t = tets_raw[i];
            // Skip tets that reference super-tet vertices
            if t.v[0] >= num_points
                || t.v[1] >= num_points
                || t.v[2] >= num_points
                || t.v[3] >= num_points
            {
                continue;
            }
            tets.push(t.v);
            adjacency.push(opp_raw[i].opp);
        }

        // Read failed verts
        let failed_count = counters.failed_count as usize;
        let failed_verts = if failed_count > 0 {
            self.buffers
                .read_failed_verts(device, queue, failed_count)
                .await
        } else {
            Vec::new()
        };

        DelaunayResult {
            tets,
            adjacency,
            failed_verts,
        }
    }

    /// Expand tetrahedron list to make room for new insertions.
    /// Port of GpuDel::expandTetraList() from GpuDelaunay.cu:457-606
    ///
    /// Allocates space for new tets that will be created during insertion.
    /// Expands by `num_new_verts * MEAN_VERTEX_DEGREE` slots.
    pub fn expand_tetra_list(
        &mut self,
        _encoder: &mut wgpu::CommandEncoder,
        _queue: &wgpu::Queue,
        num_new_verts: u32,
    ) {
        let old_tet_num = self.current_tet_num;
        let ins_extra_space = num_new_verts * MEAN_VERTEX_DEGREE;
        let new_tet_num = old_tet_num + ins_extra_space;

        eprintln!(
            "[EXPAND] Expanding tet list: {} → {} (+{} verts × {} = {} tets)",
            old_tet_num, new_tet_num, num_new_verts, MEAN_VERTEX_DEGREE, ins_extra_space
        );

        // Check if we have enough pre-allocated space
        // CRITICAL: WGPU buffers are fixed size (unlike CUDA's dynamic .grow())
        if new_tet_num > self.max_tets {
            eprintln!(
                "[EXPAND] WARNING: Requested {} tets exceeds max_tets {}!",
                new_tet_num, self.max_tets
            );
            eprintln!("[EXPAND] Clamping to max_tets. Some insertions may fail.");
            self.current_tet_num = self.max_tets;
            return;
        }

        // Update current capacity
        self.current_tet_num = new_tet_num;

        eprintln!("[EXPAND] ✓ Expansion complete, current_tet_num = {}", self.current_tet_num);

        // Note: WGPU uses pre-allocated buffers, so no reordering/shifting needed.
        // CUDA's sorting path (lines 477-556 in GpuDelaunay.cu) is not required for
        // the minimal WGPU design. See STUB_DISPATCH_ANALYSIS.md for details.
    }
}
