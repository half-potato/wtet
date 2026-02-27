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
        let num_points = points.len() as u32;
        // For block-based allocation, need enough tets to fill all vertex blocks
        // Each vertex (including 4 super-tet vertices) gets MEAN_VERTEX_DEGREE slots
        let min_tets_for_blocks = (num_points + 4) * MEAN_VERTEX_DEGREE;
        // Also consider typical Delaunay tet count (~6.5x points)
        let typical_tets = num_points * 10;
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

        let buffers = GpuBuffers::new(device, &gpu_points, num_points, max_tets);
        let pipelines = Pipelines::new(device, &buffers, num_points, max_tets);

        // All real points start as uninserted
        let uninserted: Vec<u32> = (0..num_points).collect();

        Self {
            buffers,
            pipelines,
            num_points,
            max_tets,
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
}
