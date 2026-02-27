use bytemuck;
use wgpu::util::DeviceExt;

use crate::types::{*, MEAN_VERTEX_DEGREE};

/// All GPU buffers used by the Delaunay algorithm.
pub struct GpuBuffers {
    /// Points: vec4<f32> × (N + 4 super-tet verts)
    pub points: wgpu::Buffer,
    /// Tet vertex indices: vec4<u32> × max_tets
    pub tets: wgpu::Buffer,
    /// Adjacency: vec4<u32> × max_tets
    pub tet_opp: wgpu::Buffer,
    /// Per-tet flags: u32 × max_tets
    pub tet_info: wgpu::Buffer,
    /// Tet containing each uninserted point: u32 × N
    pub vert_tet: wgpu::Buffer,
    /// Vote buffer: i32 × max_tets
    pub tet_vote: wgpu::Buffer,
    /// Free tet slot stack: u32 × max_tets
    pub free_stack: wgpu::Buffer,
    /// Block-based free list: u32 × max_tets (for future use)
    pub free_arr: wgpu::Buffer,
    /// Per-vertex free counts: u32 × (num_points + 4) (for future use)
    pub vert_free_arr: wgpu::Buffer,
    /// Atomic counters: u32 × 8
    pub counters: wgpu::Buffer,
    /// Uninserted point indices: u32 × N
    pub uninserted: wgpu::Buffer,
    /// Insert list: vec2<u32> × N (tet_idx, vert_idx pairs)
    pub insert_list: wgpu::Buffer,
    /// Tet to vertex mapping: u32 × max_tets (INVALID if not splitting)
    pub tet_to_vert: wgpu::Buffer,
    /// Tet split mapping: vec4<u32> × max_tets (maps old_tet -> 4 new tets)
    pub tet_split_map: wgpu::Buffer,
    /// Flip queue: u32 × (max_tets)
    pub flip_queue: wgpu::Buffer,
    /// Flip queue (double buffer): u32 × (max_tets)
    pub flip_queue_next: wgpu::Buffer,
    /// Flip count: atomic u32 × 1
    pub flip_count: wgpu::Buffer,
    /// Failed vertices output: u32 × N
    pub failed_verts: wgpu::Buffer,
    /// Prefix sum scratch: u32 × max_tets
    pub prefix_sum_data: wgpu::Buffer,
    /// Prefix sum block sums
    pub prefix_sum_blocks: wgpu::Buffer,
    /// Staging buffer for readback
    pub staging: wgpu::Buffer,

    pub num_points: u32,
    pub max_tets: u32,
}

impl GpuBuffers {
    pub fn new(
        device: &wgpu::Device,
        points: &[GpuPoint],
        num_points: u32,
        max_tets: u32,
    ) -> Self {
        let storage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let points_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("points"),
            contents: bytemuck::cast_slice(points),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let tets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tets"),
            size: (max_tets as u64) * 16, // vec4<u32> = 16 bytes
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let tet_opp = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tet_opp"),
            size: (max_tets as u64) * 16,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let tet_info = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tet_info"),
            size: (max_tets as u64) * 4,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let vert_tet = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vert_tet"),
            // N real points + 4 super-tet
            size: ((num_points + 4) as u64) * 4,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let tet_vote = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tet_vote"),
            size: (max_tets as u64) * 4,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        // Free stack: initially filled with indices [1..max_tets) since tet 0 is the super-tet
        let free_data: Vec<u32> = (1..max_tets).collect();
        let free_stack = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("free_stack"),
            contents: bytemuck::cast_slice(&free_data),
            usage: storage_rw,
        });

        // Initialize free_arr with block-based allocation.
        // Each vertex gets a block of MEAN_VERTEX_DEGREE tet slots.
        // Distribute available tets across vertex blocks.
        let num_vertices = (num_points + 4) as usize;
        let free_arr_size = num_vertices * MEAN_VERTEX_DEGREE as usize;
        let mut free_data = vec![0xFFFFFFFFu32; free_arr_size];

        // Distribute tets 1..max_tets-1 across vertex blocks
        // Each vertex gets a contiguous range of tets in their block
        let mut tet_idx = 1u32; // Start from 1 (tet 0 is super-tet)
        for vertex in 0..num_vertices {
            let block_start = vertex * MEAN_VERTEX_DEGREE as usize;
            let block_end = block_start + MEAN_VERTEX_DEGREE as usize;

            for slot in block_start..block_end {
                if tet_idx < max_tets {
                    free_data[slot] = tet_idx;
                    tet_idx += 1;
                } else {
                    break; // No more tets to distribute
                }
            }
        }

        let free_arr = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("free_arr"),
            contents: bytemuck::cast_slice(&free_data),
            usage: storage_rw,
        });

        // Initialize vert_free_arr: each vertex starts with MEAN_VERTEX_DEGREE free slots
        let vert_free_data = vec![MEAN_VERTEX_DEGREE; num_vertices];
        let vert_free_arr = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vert_free_arr"),
            contents: bytemuck::cast_slice(&vert_free_data),
            usage: storage_rw,
        });

        // Counters: 8 × u32 (atomic)
        let counter_init = GpuCounters {
            free_count: max_tets - 1, // all slots except 0 are free
            active_count: 0,
            inserted_count: 0,
            failed_count: 0,
            scratch: [0; 4],
        };
        let counters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("counters"),
            contents: bytemuck::bytes_of(&counter_init),
            usage: storage_rw,
        });

        let uninserted = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uninserted"),
            size: (num_points as u64) * 4,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let insert_list = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("insert_list"),
            size: (num_points as u64) * 8, // vec2<u32> = 8 bytes
            usage: storage_rw,
            mapped_at_creation: false,
        });

        // Initialize tet_to_vert to INVALID (for concurrent split detection)
        let tet_to_vert_data = vec![0xFFFFFFFFu32; max_tets as usize];
        let tet_to_vert = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tet_to_vert"),
            contents: bytemuck::cast_slice(&tet_to_vert_data),
            usage: storage_rw,
        });

        let tet_split_map = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tet_split_map"),
            size: (max_tets as u64) * 16, // vec4<u32> per tet
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let flip_queue = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flip_queue"),
            size: (max_tets as u64) * 4,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let flip_queue_next = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flip_queue_next"),
            size: (max_tets as u64) * 4,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let flip_count = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flip_count"),
            size: 4, // single atomic u32
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let failed_verts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("failed_verts"),
            size: (num_points as u64) * 4,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let prefix_sum_data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prefix_sum_data"),
            size: (max_tets as u64) * 4,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let num_blocks = (max_tets + 511) / 512;
        let prefix_sum_blocks = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prefix_sum_blocks"),
            size: (num_blocks as u64) * 4,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        // Staging buffer: large enough for the biggest readback we'll do
        let max_readback = (max_tets as u64) * 16; // tets are the biggest
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: max_readback,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            points: points_buf,
            tets,
            tet_opp,
            tet_info,
            vert_tet,
            tet_vote,
            free_stack,
            free_arr,
            vert_free_arr,
            counters,
            uninserted,
            insert_list,
            tet_to_vert,
            tet_split_map,
            flip_queue,
            flip_queue_next,
            flip_count,
            failed_verts,
            prefix_sum_data,
            prefix_sum_blocks,
            staging,
            num_points,
            max_tets,
        }
    }

    /// Upload the uninserted vertex list to the GPU.
    pub fn upload_uninserted(&self, queue: &wgpu::Queue, indices: &[u32]) {
        queue.write_buffer(&self.uninserted, 0, bytemuck::cast_slice(indices));
    }

    /// Read counters back to CPU.
    pub async fn read_counters(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> GpuCounters {
        let size = std::mem::size_of::<GpuCounters>() as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("counter_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.counters, 0, &staging, 0, size);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().unwrap();

        let data = slice.get_mapped_range();
        let counters: GpuCounters = *bytemuck::from_bytes(&data[..size as usize]);
        drop(data);
        staging.unmap();
        counters
    }

    /// Read tets back to CPU.
    pub async fn read_tets(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        count: usize,
    ) -> Vec<GpuTet> {
        self.read_buffer_as(device, queue, &self.tets, count).await
    }

    /// Read adjacency back to CPU.
    pub async fn read_opp(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        count: usize,
    ) -> Vec<GpuTetOpp> {
        self.read_buffer_as(device, queue, &self.tet_opp, count).await
    }

    /// Read failed verts back to CPU.
    pub async fn read_failed_verts(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        count: usize,
    ) -> Vec<u32> {
        self.read_buffer_as(device, queue, &self.failed_verts, count)
            .await
    }

    /// Generic buffer readback.
    pub async fn read_buffer_as<T: bytemuck::Pod>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Vec<T> {
        let elem_size = std::mem::size_of::<T>() as u64;
        let size = (count as u64) * elem_size;
        if size == 0 {
            return Vec::new();
        }

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data[..size as usize]).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    /// Read flip_count (single u32) back to CPU.
    pub async fn read_flip_count(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> u32 {
        let vals: Vec<u32> = self.read_buffer_as(device, queue, &self.flip_count, 1).await;
        vals[0]
    }

    /// Write a uniform params buffer (vec4<u32>).
    pub fn create_params_buffer(device: &wgpu::Device, params: [u32; 4]) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }
}
