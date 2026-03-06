pub mod buffers;
pub mod pipelines;
pub mod dispatch;

use std::collections::HashMap;
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
    /// Vote offset for separating insertion votes from flip votes
    /// CUDA Reference: GpuDelaunay.cu:1121-1128
    pub vote_offset: u32,
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
        // Each vertex (including 4 super-tet + 1 infinity) gets MEAN_VERTEX_DEGREE slots
        // Plus MEAN_VERTEX_DEGREE slots for the infinity block (used during flips)
        let min_tets_for_blocks = (num_points + 5) * MEAN_VERTEX_DEGREE + MEAN_VERTEX_DEGREE;
        // Also consider typical Delaunay tet count (~6.5x points) + overhead for retries
        // Allocate generously: 30x points to handle worst-case insertions + flips + retries
        let typical_tets = num_points * 30;
        let max_tets = min_tets_for_blocks.max(typical_tets).max(64);

        // Build point buffer: N real points + 4 super-tet vertices + 1 infinity
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

        // Infinity vertex (used for boundary faces in 5-tet initialization)
        // Placed even farther out than super-tet vertices
        gpu_points.push(GpuPoint::new(10.0 * big, 10.0 * big, 10.0 * big)); // n+4 (inf_idx)

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
            current_tet_num: 5, // Start with 5 (5-tet topology created by init kernel)
            uninserted,
            vote_offset: max_tets, // Initialize to max (CUDA uses INT_MAX, we use max_tets)
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

        // Phase 1: Filter alive tets and build buffer_idx → output_idx mapping
        let inf_idx = num_points + 4; // Infinity vertex (after 4 super-tet vertices)
        let mut tets = Vec::new();
        let mut idx_map: HashMap<usize, usize> = HashMap::new();

        for i in 0..max {
            if (info_raw[i].flags & TET_ALIVE) == 0 {
                continue;
            }
            let t = tets_raw[i];
            // Skip tets containing infinity vertex (internal boundary representation)
            // Keep tets with super-tet vertices (these form the actual convex hull)
            if t.v[0] == inf_idx
                || t.v[1] == inf_idx
                || t.v[2] == inf_idx
                || t.v[3] == inf_idx
            {
                continue;
            }
            idx_map.insert(i, tets.len()); // buffer_idx → output_idx
            tets.push(t.v);
        }

        // Phase 2: Remap adjacency (buffer indices → output indices)
        let mut adjacency = Vec::new();
        for i in 0..max {
            if (info_raw[i].flags & TET_ALIVE) == 0 {
                continue;
            }
            let t = tets_raw[i];
            // Skip tets containing infinity vertex (same filter as phase 1)
            if t.v[0] == inf_idx
                || t.v[1] == inf_idx
                || t.v[2] == inf_idx
                || t.v[3] == inf_idx
            {
                continue;
            }

            let opp = &opp_raw[i].opp;
            let mut remapped = [INVALID; 4];
            for f in 0..4 {
                if opp[f] == INVALID {
                    continue;
                }
                let (buf_idx, face) = decode_opp(opp[f]);
                // Remap buffer index to output index (or leave INVALID if neighbor was filtered)
                if let Some(&out_idx) = idx_map.get(&(buf_idx as usize)) {
                    remapped[f] = encode_opp(out_idx as u32, face);
                }
            }
            adjacency.push(remapped);
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
        queue: &wgpu::Queue,
        num_new_verts: u32,
        insert_list: &[[u32; 2]], // (tet_idx, position) pairs
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

        // Update free_arr for vertices being inserted
        // Each vertex V gets MEAN_VERTEX_DEGREE tets starting at old_tet_num + (position * MEAN_VERTEX_DEGREE)
        let mut free_arr_updates: Vec<(usize, u32)> = Vec::new(); // (index, value) pairs

        for (idx, &[_tet_idx, position]) in insert_list.iter().enumerate() {
            let vertex_id = self.uninserted[position as usize];
            let tet_base = old_tet_num + (idx as u32 * MEAN_VERTEX_DEGREE);
            let free_arr_base = (vertex_id * MEAN_VERTEX_DEGREE) as usize;

            // Update vertex's free list with its allocated tets
            for slot in 0..MEAN_VERTEX_DEGREE {
                free_arr_updates.push((free_arr_base + slot as usize, tet_base + slot));
            }

            eprintln!("[EXPAND] Vertex {} gets tets [{}-{}]", vertex_id, tet_base, tet_base + MEAN_VERTEX_DEGREE - 1);
        }

        // Write updates to GPU buffer
        for (index, value) in free_arr_updates {
            queue.write_buffer(
                &self.buffers.free_arr,
                (index * 4) as u64,
                bytemuck::cast_slice(&[value]),
            );
        }

        // Update current capacity
        self.current_tet_num = new_tet_num;

        eprintln!("[EXPAND] ✓ Expansion complete, current_tet_num = {}", self.current_tet_num);

        // Note: WGPU uses pre-allocated buffers, so no reordering/shifting needed.
        // CUDA's sorting path (lines 477-556 in GpuDelaunay.cu) is not required for
        // the minimal WGPU design. See STUB_DISPATCH_ANALYSIS.md for details.
    }
}
