pub mod buffers;
pub mod pipelines;
pub mod dispatch;

pub use dispatch::FlipCompactMode;

use std::collections::HashMap;
use crate::types::{*, MEAN_VERTEX_DEGREE};
use crate::profiler::{GpuProfiler, CpuProfiler};
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
    /// Number of points that still need to be inserted.
    /// The uninserted vertex IDs are stored in GPU buffer and compacted in place.
    pub num_uninserted: u32,
    /// Vote offset for separating insertion votes from flip votes
    /// CUDA Reference: GpuDelaunay.cu:1121-1128
    pub vote_offset: u32,
    /// Maximum number of flips before overflow (matches flip_arr allocation)
    pub max_flips: u32,
    /// Enable dynamic partial binding for large datasets (> 128 MB)
    pub use_partial_binding: bool,
    /// Device reference for creating dynamic bind groups
    pub device: wgpu::Device,
    /// GPU timestamp period (nanoseconds per tick)
    pub timestamp_period: f32,
    /// GPU profiler for timing GPU operations
    pub gpu_profiler: Option<GpuProfiler>,
    /// CPU profiler for timing CPU-side operations
    pub cpu_profiler: CpuProfiler,
    /// Mapping from sorted index to original index (used when Morton sorting is enabled).
    /// If None, points were not reordered.
    /// If Some(vec), then sorted_idx i corresponds to original point vec[i].
    pub point_mapping: Option<Vec<u32>>,
}

impl GpuState {
    pub async fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        points: &[[f32; 3]],
        config: &GDelConfig,
    ) -> Self {
        eprintln!("\n[GPU STATE] Initializing GpuState...");
        let num_points = points.len() as u32;
        eprintln!("[GPU STATE] Input: {} points", num_points);

        // Sort points by Morton code if enabled (reduces iteration count by improving spatial locality)
        // CUDA Reference: GpuDelaunay.cu:661-684
        let (sorted_points, point_mapping) = if config.enable_sorting {
            eprintln!("[GPU STATE] Morton code sorting enabled");
            let (sorted, mapping) = sort_points_by_morton(points);
            eprintln!("[GPU STATE] ✓ Points sorted by Morton code");

            // Debug: Show that sorting actually reordered points
            let mut reordered = 0;
            for i in 0..mapping.len().min(100) {
                if mapping[i] != i as u32 {
                    reordered += 1;
                }
            }
            eprintln!("[GPU STATE] First 100 points: {} reordered, {} unchanged",
                     reordered, 100 - reordered);

            (sorted, Some(mapping))
        } else {
            eprintln!("[GPU STATE] Morton code sorting disabled");
            (points.to_vec(), None)
        };

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
        // Use sorted_points (which may be reordered by Morton code, or same as input if sorting disabled)
        let mut gpu_points: Vec<GpuPoint> = sorted_points
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

        // All real points start as uninserted (vertex IDs 0..num_points already in GPU buffer)
        let num_uninserted = num_points;

        // Initialize profilers
        let timestamp_period = _queue.get_timestamp_period();
        let gpu_profiler = if device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            Some(GpuProfiler::new(device, 2048)) // 2048 timestamps = 1024 scopes
        } else {
            eprintln!("[PROFILER] TIMESTAMP_QUERY not supported, GPU profiling disabled");
            None
        };
        let cpu_profiler = CpuProfiler::new();

        eprintln!("[GPU STATE] ✓ GpuState initialization complete\n");

        Self {
            buffers,
            pipelines,
            num_points,
            max_tets,
            current_tet_num: 5, // Start with 5 (5-tet topology created by init kernel)
            num_uninserted,
            vote_offset: max_tets, // Initialize to max (CUDA uses INT_MAX, we use max_tets)
            max_flips: max_tets / 2, // CUDA allocates TetMax/2 flip items
            use_partial_binding,
            device: device.clone(),
            timestamp_period,
            gpu_profiler,
            cpu_profiler,
            point_mapping,
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

        // Phase 3: Remap vertex indices to original point indices (if Morton sorting was used)
        if let Some(mapping) = &self.point_mapping {
            eprintln!("[READBACK] Remapping vertex indices from sorted to original order");
            for tet in &mut tets {
                for v in tet.iter_mut() {
                    // Only remap real point indices (0..num_points)
                    // Leave super-tet (num_points..num_points+4) and infinity (num_points+4) unchanged
                    if (*v as usize) < mapping.len() {
                        *v = mapping[*v as usize];
                    }
                }
            }
        }

        // Read failed verts and remap if needed
        let failed_count = counters.failed_count as usize;
        let mut failed_verts = if failed_count > 0 {
            self.buffers
                .read_failed_verts(device, queue, failed_count)
                .await
        } else {
            Vec::new()
        };

        // Remap failed verts to original indices
        if let Some(mapping) = &self.point_mapping {
            for v in &mut failed_verts {
                if (*v as usize) < mapping.len() {
                    *v = mapping[*v as usize];
                }
            }
        }

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

        // Dispatch GPU kernel to reinitialize free lists
        self.dispatch_update_vert_free_list(
            _encoder,
            queue,
            num_new_verts,
            old_tet_num,
        );

        // Update current capacity
        self.current_tet_num = new_tet_num;

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

/// Sort points by Morton code (Z-order curve) for better spatial locality.
///
/// Returns (sorted_points, mapping) where mapping[i] is the original index
/// of the point at sorted position i.
///
/// CUDA Reference: GpuDelaunay.cu:661-684
fn sort_points_by_morton(points: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<u32>) {
    use crate::morton::compute_morton_code;

    // 1. Compute bounding box
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for p in points {
        for i in 0..3 {
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }
    }

    // 2. Compute Morton code for each point and create indexed list
    let mut indexed_points: Vec<(u32, usize, [f32; 3])> = points
        .iter()
        .enumerate()
        .map(|(i, &pt)| {
            let morton = compute_morton_code(pt, min, max);
            (morton, i, pt)
        })
        .collect();

    // 3. Sort by Morton code (stable sort preserves order for equal codes)
    indexed_points.sort_by_key(|(morton, _, _)| *morton);

    // 4. Extract sorted points and mapping
    let sorted_points: Vec<[f32; 3]> = indexed_points
        .iter()
        .map(|(_, _, pt)| *pt)
        .collect();

    let point_mapping: Vec<u32> = indexed_points
        .iter()
        .map(|(_, orig_idx, _)| *orig_idx as u32)
        .collect();

    (sorted_points, point_mapping)
}
