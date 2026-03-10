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
    /// Maximum number of flips before overflow (matches flip_arr allocation)
    pub max_flips: u32,
    /// Enable dynamic partial binding for large datasets (> 128 MB)
    pub use_partial_binding: bool,
    /// Device reference for creating dynamic bind groups
    pub device: wgpu::Device,
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
        // CUDA allocates TetMax = pointNum × 8.5
        // WGPU: use 8× to stay under 256 MB buffer allocation limit (at 2M points)
        // 2M points × 8 × 16 bytes = 256 MB (exactly at limit)
        let typical_tets = num_points * 8;
        let max_tets = min_tets_for_blocks.max(typical_tets).max(64);

        // Validate buffer sizes against WGPU limits
        let tet_buffer_size = (max_tets as u64) * 16; // vec4<u32> = 16 bytes
        let flip_arr_size = ((max_tets / 2) as u64) * 32; // FlipItem = 32 bytes

        // WGPU has two limits:
        // 1. Buffer allocation limit: 256 MB per buffer
        // 2. Buffer binding limit: 128 MB per binding (in bind groups)
        let limit_256mb = 256 * 1024 * 1024;
        let limit_128mb = 128 * 1024 * 1024;

        if tet_buffer_size > limit_256mb {
            panic!(
                "Dataset too large: {} points requires {} MB for tet buffers.\n\
                 WGPU buffer allocation limit is 256 MB per buffer.\n\
                 Maximum supported: ~2M points. Please chunk your input into smaller batches.",
                num_points, tet_buffer_size / (1024 * 1024)
            );
        }

        // Warn about binding size limit (requires partial binding for all pipelines)
        if tet_buffer_size > limit_128mb {
            eprintln!("[WARNING] Dataset size ({} points) exceeds 128 MB binding limit.", num_points);
            eprintln!("[WARNING] Some pipelines may fail during bind group creation.");
            eprintln!("[WARNING] For guaranteed support, use < 1M points.");
            eprintln!("[WARNING] 2M+ points requires complex multi-buffer chunking (not yet implemented).");
        }

        // Enable partial binding for datasets > 128 MB
        let use_partial_binding = flip_arr_size > 128 * 1024 * 1024;
        if use_partial_binding {
            eprintln!("[GPU STATE] Large dataset detected: using partial buffer binding");
            eprintln!("[GPU STATE] flip_arr size: {:.2} MB",
                     flip_arr_size as f64 / (1024.0 * 1024.0));
        }

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
            max_flips: max_tets / 2, // CUDA allocates TetMax/2 flip items
            use_partial_binding,
            device: device.clone(),
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
            // NEW: Filter out tets with all-zero or invalid vertices
            // (Prevents uninitialized tets from passing through)
            if t.v[0] == 0 && t.v[1] == 0 && t.v[2] == 0 && t.v[3] == 0 {
                eprintln!("[READBACK] Skipping invalid tet {} with zero vertices", i);
                continue;
            }
            // Also filter out tets with out-of-range vertices
            if t.v[0] > num_points + 4
                || t.v[1] > num_points + 4
                || t.v[2] > num_points + 4
                || t.v[3] > num_points + 4
            {
                eprintln!(
                    "[READBACK] Skipping tet {} with out-of-range vertices {:?}",
                    i, t.v
                );
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

        // Update free_arr for vertices being inserted via GPU kernel
        // Port of CUDA kerUpdateVertFreeList (KerDivision.cu:1126-1149)
        // This reinitializes the free lists to mask the allocation/donation bug (see CUDA_FLAWS.md)

        // Print what we're allocating (for debugging)
        for (idx, &[_tet_idx, position]) in insert_list.iter().enumerate() {
            let vertex_id = self.uninserted[position as usize];
            let tet_base = old_tet_num + (idx as u32 * MEAN_VERTEX_DEGREE);
            eprintln!("[EXPAND] Vertex {} gets tets [{}-{}]", vertex_id, tet_base, tet_base + MEAN_VERTEX_DEGREE - 1);
        }

        // Dispatch GPU kernel to reinitialize free lists
        eprintln!("[EXPAND] Dispatching update_vert_free_list: {} vertices, start_idx={}", num_new_verts, old_tet_num);
        self.dispatch_update_vert_free_list(
            _encoder,
            queue,
            num_new_verts,
            old_tet_num,
        );
        eprintln!("[EXPAND] ✓ Free lists reinitialized");

        // Update current capacity
        self.current_tet_num = new_tet_num;

        eprintln!("[EXPAND] ✓ Expansion complete, current_tet_num = {}", self.current_tet_num);

        // Note: WGPU uses pre-allocated buffers, so no reordering/shifting needed.
        // CUDA's sorting path (lines 477-556 in GpuDelaunay.cu) is not required for
        // the minimal WGPU design. See STUB_DISPATCH_ANALYSIS.md for details.
    }

    /// Dispatch kerUpdateVertFreeList - reinitialize free lists for newly inserted vertices
    /// Port of CUDA kerUpdateVertFreeList (KerDivision.cu:1126-1149)
    ///
    /// CRITICAL: This is called EVERY iteration to mask the allocation/donation bug.
    /// See CUDA_FLAWS.md for details on why this is necessary.
    fn dispatch_update_vert_free_list(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        num_inserted: u32,
        start_free_idx: u32,
    ) {
        // Update params: [num_inserted, start_free_idx, 0, 0]
        queue.write_buffer(
            &self.pipelines.update_vert_free_params,
            0,
            bytemuck::cast_slice(&[num_inserted, start_free_idx, 0u32, 0u32]),
        );

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("update_vert_free_list"),
            timestamp_writes: None,
        });

        cpass.set_pipeline(&self.pipelines.update_vert_free_pipeline);
        cpass.set_bind_group(0, &self.pipelines.update_vert_free_bind_group, &[]);

        // Dispatch: num_inserted * MEAN_VERTEX_DEGREE threads
        let num_threads = num_inserted * MEAN_VERTEX_DEGREE;
        let workgroups = (num_threads + 255) / 256;  // Workgroup size = 256
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }
}
