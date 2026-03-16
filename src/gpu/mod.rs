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

        // Sort points for better spatial locality (reduces voting conflicts during insertion)
        // Priority: Hilbert (stratified) > Morton > Sequential
        // CUDA Reference: GpuDelaunay.cu:661-684
        let (sorted_points, point_mapping) = if config.enable_hilbert_sorting {
            eprintln!("[GPU STATE] Hilbert curve sorting enabled (stratified sampling)");
            let (sorted, mapping) = sort_points_by_hilbert(points);
            eprintln!("[GPU STATE] ✓ Points sorted by Hilbert curve with stratified sampling");

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
        } else if config.enable_sorting {
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
            eprintln!("[GPU STATE] Sequential insertion order (no reordering)");
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
            // Keep ALL alive tets including infinity and super-tet tets.
            // The CUDA code's compactTetras keeps all alive tets — removing infinity
            // tets breaks the closed manifold needed by star splaying.
            // Quality checks skip super-tet/infinity tets via the num_real_points filter.

            // Filter out tets with all-zero or invalid vertices
            // (Prevents uninitialized tets from passing through)
            if t.v[0] == 0 && t.v[1] == 0 && t.v[2] == 0 && t.v[3] == 0 {
                continue;
            }
            // Also filter out tets with out-of-range vertices
            if t.v[0] > num_points + 4
                || t.v[1] > num_points + 4
                || t.v[2] > num_points + 4
                || t.v[3] > num_points + 4
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
            // Same filters as phase 1
            if t.v[0] == 0 && t.v[1] == 0 && t.v[2] == 0 && t.v[3] == 0 {
                continue;
            }
            if t.v[0] > num_points + 4
                || t.v[1] > num_points + 4
                || t.v[2] > num_points + 4
                || t.v[3] > num_points + 4
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

/// Sort points by Hilbert curve index, then apply strided sampling.
///
/// Strategy:
/// 1. Sort points by Hilbert curve index (establishes spatial ordering)
/// 2. Apply strided sampling to distribute insertions across the curve
///
/// Strided sampling inserts every Nth point in the first pass, ensuring
/// good spatial distribution and reduced voting conflicts.
///
/// Returns (sorted_points, mapping) where mapping[i] is the original index
/// of the point at sorted position i.
fn sort_points_by_hilbert(points: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<u32>) {
    use crate::hilbert::compute_hilbert_index;

    // 1. Compute bounding box
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for p in points {
        for i in 0..3 {
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }
    }

    // 2. Compute Hilbert indices using established algorithm (hilbert_index crate)
    let mut indexed_points: Vec<(usize, usize, [f32; 3])> = points
        .iter()
        .enumerate()
        .map(|(i, &pt)| {
            let hilbert_idx = compute_hilbert_index(pt, min, max);
            (hilbert_idx, i, pt)
        })
        .collect();

    // 3. Sort by Hilbert index (stable sort preserves order for equal codes)
    indexed_points.sort_by_key(|(hilbert_idx, _, _)| *hilbert_idx);

    // 4. Apply stratified (hierarchical) sampling to reorder the Hilbert-sorted points
    // Start with wide stride (~0.1% sampling), progressively narrow to fill gaps
    // For 2M points: sqrt(2M) ≈ 1414 stride → ~1400 points in first pass (~0.07%)
    let n = indexed_points.len();
    let initial_stride = calculate_initial_stride(n);
    eprintln!("[GPU STATE] Stratified sampling: {} points, initial stride = {} (~{:.2}% first pass)",
              n, initial_stride, 100.0 / initial_stride as f64);
    let strided_points = apply_strided_sampling_indexed(&indexed_points, initial_stride);

    // 5. Extract final points and mapping
    let sorted_points: Vec<[f32; 3]> = strided_points
        .iter()
        .map(|(_, _, pt)| *pt)
        .collect();

    let point_mapping: Vec<u32> = strided_points
        .iter()
        .map(|(_, orig_idx, _)| *orig_idx as u32)
        .collect();

    (sorted_points, point_mapping)
}

/// Calculate initial stride for stratified sampling based on point count.
///
/// Uses aggressive square root scaling to ensure very sparse initial sampling:
/// - 100 points: stride 64 → 1 point (~1.5%)
/// - 1K points: stride 64 → 15 points (~1.5%)
/// - 10K points: stride 128 → 78 points (~0.8%)
/// - 100K points: stride 512 → 195 points (~0.2%)
/// - 1M points: stride 2048 → 488 points (~0.05%)
/// - 2M points: stride 4096 → 488 points (~0.02%)
///
/// For large datasets (>100K), this ensures first pass inserts <0.1% of points.
fn calculate_initial_stride(num_points: usize) -> usize {
    // Aggressive scaling: sqrt(N) * 1.5 for large datasets
    let sqrt_n = (num_points as f64).sqrt();
    let scaled = if num_points > 100_000 {
        sqrt_n * 2.0  // 2x more aggressive for large datasets
    } else {
        sqrt_n
    };
    let stride = scaled.max(64.0).min(16384.0);

    // Round UP to nearest power of 2 for clean halving pattern and sparser sampling
    let log2 = stride.log2().ceil();
    let stride_pow2 = (2.0_f64).powf(log2) as usize;

    stride_pow2
}

/// Apply stratified (hierarchical) sampling to reorder points.
///
/// Strategy: Insert points in multiple passes with progressively narrowing strides.
/// This ensures good spatial distribution at all scales.
///
/// Example with initial_stride=128:
/// - Pass 1 (stride 128): [0, 128, 256, 384, ...]        <- ~1% of points
/// - Pass 2 (stride 64):  [64, 192, 320, 448, ...]       <- ~1% more
/// - Pass 3 (stride 32):  [32, 96, 160, 224, 288, ...]   <- ~2% more
/// - Pass 4 (stride 16):  [16, 48, 80, 112, 144, ...]    <- ~4% more
/// - Pass 5 (stride 8):   [8, 24, 40, 56, 72, ...]       <- ~8% more
/// - Pass 6 (stride 4):   [4, 12, 20, 28, 36, ...]       <- ~16% more
/// - Pass 7 (stride 2):   [2, 6, 10, 14, 18, 22, ...]    <- ~32% more
/// - Pass 8 (stride 1):   [1, 3, 5, 7, 9, 11, ...]       <- remaining ~35%
///
/// This creates a multi-scale spatial distribution that reduces voting conflicts.
///
/// # Arguments
/// * `indexed_points` - Points sorted by Hilbert index
/// * `initial_stride` - Starting stride (e.g., 128 for ~1% initial sampling)
///
/// # Returns
/// Reordered points with stratified sampling applied
fn apply_strided_sampling_indexed(
    indexed_points: &[(usize, usize, [f32; 3])],
    initial_stride: usize,
) -> Vec<(usize, usize, [f32; 3])> {
    let n = indexed_points.len();
    let mut result = Vec::with_capacity(n);
    let mut inserted = vec![false; n];

    // Start with widest stride, progressively halve
    let mut stride = initial_stride;
    while stride >= 1 {
        // Insert points at indices [0, stride, 2*stride, 3*stride, ...]
        // that haven't been inserted yet
        let mut i = 0;
        while i < n {
            if !inserted[i] {
                result.push(indexed_points[i]);
                inserted[i] = true;
            }
            i += stride;
        }

        // Halve stride for next pass (progressively fill gaps)
        stride /= 2;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stratified_sampling_pattern() {
        // Create test data: 16 points with indices 0-15
        let indexed: Vec<(usize, usize, [f32; 3])> = (0..16)
            .map(|i| (i, i, [i as f32, 0.0, 0.0]))
            .collect();

        // Apply stratified sampling with initial stride 8
        let strided = apply_strided_sampling_indexed(&indexed, 8);

        // Extract just the indices for verification
        let indices: Vec<usize> = strided.iter().map(|(_, idx, _)| *idx).collect();

        // Expected pattern with initial stride 8 (progressively halving):
        // Pass 1 (stride 8): 0, 8           <- ~12% of points (2/16)
        // Pass 2 (stride 4): 4, 12          <- ~12% more (2/16)
        // Pass 3 (stride 2): 2, 6, 10, 14   <- ~25% more (4/16)
        // Pass 4 (stride 1): 1, 3, 5, 7, 9, 11, 13, 15 <- remaining ~50% (8/16)
        let expected = vec![
            0, 8,               // Wide spacing (stride 8)
            4, 12,              // Half the spacing (stride 4)
            2, 6, 10, 14,       // Half again (stride 2)
            1, 3, 5, 7, 9, 11, 13, 15, // Fill remaining (stride 1)
        ];

        assert_eq!(indices, expected,
            "Stratified sampling should produce hierarchical pattern [0,8, 4,12, 2,6,10,14, ...]");
    }

    #[test]
    fn test_stratified_sampling_preserves_count() {
        let indexed: Vec<(usize, usize, [f32; 3])> = (0..100)
            .map(|i| (i, i, [i as f32, 0.0, 0.0]))
            .collect();

        let strided = apply_strided_sampling_indexed(&indexed, 64);

        assert_eq!(strided.len(), 100, "Stratified sampling should preserve point count");

        // Verify all original indices are present
        let mut indices: Vec<usize> = strided.iter().map(|(_, idx, _)| *idx).collect();
        indices.sort();
        let expected: Vec<usize> = (0..100).collect();
        assert_eq!(indices, expected, "All original indices should be present");
    }

    #[test]
    fn test_calculate_initial_stride() {
        // Verify stride calculation scales appropriately
        assert_eq!(calculate_initial_stride(100), 64);        // Small: ~1.5%
        assert_eq!(calculate_initial_stride(1_000), 64);      // 1K: ~1.5%
        assert_eq!(calculate_initial_stride(10_000), 128);    // 10K: ~0.8%
        assert_eq!(calculate_initial_stride(100_000), 512);   // 100K: ~0.2%
        assert_eq!(calculate_initial_stride(1_000_000), 2048); // 1M: ~0.05%
        assert_eq!(calculate_initial_stride(2_000_000), 4096); // 2M: ~0.02%

        // Edge cases
        assert_eq!(calculate_initial_stride(10), 64);       // Min clamp
        assert_eq!(calculate_initial_stride(1_000_000_000), 16384); // Max clamp

        // Verify first pass percentages are reasonable
        let stride_2m = calculate_initial_stride(2_000_000);
        let first_pass = 2_000_000 / stride_2m;
        assert!(first_pass < 1000, "2M points should insert <1000 in first pass (got {})", first_pass);
    }

    #[test]
    fn test_stratified_sampling_order() {
        // Test that first few points are widely spaced
        let indexed: Vec<(usize, usize, [f32; 3])> = (0..256)
            .map(|i| (i, i, [i as f32, 0.0, 0.0]))
            .collect();

        let strided = apply_strided_sampling_indexed(&indexed, 128);
        let indices: Vec<usize> = strided.iter().map(|(_, idx, _)| *idx).collect();

        // First pass (stride 128): should insert indices [0, 128]
        assert_eq!(indices[0], 0, "First point should be at index 0");
        assert_eq!(indices[1], 128, "Second point should be at index 128");

        // Second pass (stride 64): should insert indices [64, 192]
        assert_eq!(indices[2], 64, "Third point should be at index 64");
        assert_eq!(indices[3], 192, "Fourth point should be at index 192");

        // Verify hierarchical refinement continues
        // After 2 passes with stride 128→64, we should have 4 points: [0, 128, 64, 192]
        assert!(indices[0..4].contains(&0));
        assert!(indices[0..4].contains(&64));
        assert!(indices[0..4].contains(&128));
        assert!(indices[0..4].contains(&192));
    }
}
