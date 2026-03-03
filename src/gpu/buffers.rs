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

    // --- New buffers for missing kernels ---
    /// FlipItem array: vec4<i32> × (max_tets * 2) - stored as pairs of vec4
    pub flip_arr: wgpu::Buffer,
    /// Tet message array for concurrent flip detection: vec2<i32> × max_tets
    pub tet_msg_arr: wgpu::Buffer,
    /// Encoded face vertex indices: i32 × max_tets
    pub encoded_face_vi_arr: wgpu::Buffer,
    /// Tet to flip trace chain: i32 × max_tets
    pub tet_to_flip: wgpu::Buffer,
    /// Scatter array for reordering: u32 × (num_points + 4)
    pub scatter_arr: wgpu::Buffer,
    /// Order array for reordering: u32 × max_tets
    pub order_arr: wgpu::Buffer,
    /// Inserted vertices array: u32 × max_tets
    pub ins_vert_vec: wgpu::Buffer,
    /// Reverse mapping array: u32 × max_tets
    pub rev_map_arr: wgpu::Buffer,
    /// Active tet vector for flipping: i32 × max_tets
    pub act_tet_vec: wgpu::Buffer,
    /// Vote array for flips: i32 × max_tets
    pub vote_arr: wgpu::Buffer,
    /// Flip to tet mapping (compacted output): i32 × max_tets
    pub flip_to_tet: wgpu::Buffer,
    /// New slot allocation for 2-3 flips: i32 × max_tets
    pub flip23_new_slot: wgpu::Buffer,

    // Debug buffers
    /// Breadcrumbs: u32 per thread (marks progress through shader)
    pub breadcrumbs: wgpu::Buffer,
    /// Per-thread debug slots: 16 * vec4<u32> per thread
    pub thread_debug: wgpu::Buffer,
    /// Debug buffer for update_uninserted_vert_tet kernel: 4 * vec4<u32> per thread
    pub update_debug: wgpu::Buffer,

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
        eprintln!("[BUFFERS] Creating GpuBuffers:");
        eprintln!("  num_points={}, max_tets={}", num_points, max_tets);
        eprintln!("  MEAN_VERTEX_DEGREE={}", MEAN_VERTEX_DEGREE);

        let storage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        eprintln!("  Creating points buffer: {} points", points.len());
        let points_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("points"),
            contents: bytemuck::cast_slice(points),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        eprintln!("  ✓ points: {} bytes", points.len() * 16);

        eprintln!("  Creating core tet buffers (max_tets={}):", max_tets);
        let tets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tets"),
            size: (max_tets as u64) * 16, // vec4<u32> = 16 bytes
            usage: storage_rw,
            mapped_at_creation: false,
        });
        eprintln!("    ✓ tets: {} bytes", max_tets * 16);

        let tet_opp = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tet_opp"),
            size: (max_tets as u64) * 16,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        eprintln!("    ✓ tet_opp: {} bytes", max_tets * 16);

        let tet_info = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tet_info"),
            size: (max_tets as u64) * 4,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        eprintln!("    ✓ tet_info: {} bytes", max_tets * 4);

        let vert_tet = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vert_tet"),
            // N real points + 4 super-tet
            size: ((num_points + 4) as u64) * 4,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        eprintln!("    ✓ vert_tet: {} bytes (num_points+4)", (num_points + 4) * 4);

        let tet_vote = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tet_vote"),
            size: (max_tets as u64) * 4,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        eprintln!("    ✓ tet_vote: {} bytes", max_tets * 4);

        // Free stack: initially filled with indices [1..max_tets) since tet 0 is the super-tet
        let free_data: Vec<u32> = (1..max_tets).collect();
        let free_stack = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("free_stack"),
            contents: bytemuck::cast_slice(&free_data),
            usage: storage_rw,
        });

        // Initialize free_arr with block-based allocation (per-vertex blocks)
        // Each vertex gets MEAN_VERTEX_DEGREE (8) tet slots
        // Vertex V's slots are at free_arr[V*8 .. V*8+7]
        // Port of CUDA's block allocation scheme (see kerSplitTetra line 120, kerUpdateBlockVertFreeList)
        let free_arr_size = max_tets as usize;
        let mut free_data = vec![0u32; free_arr_size];

        // Calculate how many vertices can be supported with current max_tets
        let max_vertex_blocks = max_tets / MEAN_VERTEX_DEGREE;

        // Initialize free slots for each vertex block
        // Vertex V gets tet indices [V*8, V*8+1, ..., V*8+7]
        for v in 0..max_vertex_blocks {
            let base = v * MEAN_VERTEX_DEGREE;
            for slot in 0..MEAN_VERTEX_DEGREE {
                free_data[(base + slot) as usize] = base + slot;
            }
        }

        eprintln!("[BUFFERS] free_arr: block-based allocation for {} vertex blocks ({} tets total)",
                  max_vertex_blocks, max_tets);

        // Initialize vert_free_arr: per-vertex free slot counts
        // Each vertex (including 4 super-tet vertices) gets MEAN_VERTEX_DEGREE slots initially
        let num_vertices = (num_points + 4) as usize;
        let mut vert_free_data = vec![MEAN_VERTEX_DEGREE; num_vertices];

        eprintln!("[BUFFERS] vert_free_arr: num_vertices={}, each with {} slots",
                  num_vertices, MEAN_VERTEX_DEGREE);

        let free_arr = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("free_arr"),
            contents: bytemuck::cast_slice(&free_data),
            usage: storage_rw,
        });
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

        // Debug buffers: breadcrumbs (u32 per thread) and debug slots (16 * vec4 per thread)
        let max_threads = max_tets; // Conservative: as many threads as tets
        let breadcrumbs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("breadcrumbs"),
            size: (max_threads as u64) * 4, // u32 per thread
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let thread_debug = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("thread_debug"),
            size: (max_threads as u64) * 16 * 16, // 16 vec4<u32> per thread
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let update_debug = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("update_debug"),
            size: (max_threads as u64) * 4 * 16, // 4 vec4<u32> per thread
            usage: storage_rw,
            mapped_at_creation: false,
        });

        // --- New buffers for missing kernels ---
        let flip_arr = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flip_arr"),
            size: (max_tets as u64) * 2 * 16, // FlipItem = 2 * vec4<i32> = 32 bytes
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let tet_msg_arr = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tet_msg_arr"),
            size: (max_tets as u64) * 8, // vec2<i32> = 8 bytes
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let encoded_face_vi_arr = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("encoded_face_vi_arr"),
            size: (max_tets as u64) * 4, // i32 = 4 bytes
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let tet_to_flip = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tet_to_flip"),
            size: (max_tets as u64) * 4, // i32 = 4 bytes
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let scatter_arr = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scatter_arr"),
            size: ((num_points + 4) as u64) * 4, // u32 = 4 bytes
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let order_arr = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("order_arr"),
            size: (max_tets as u64) * 4, // u32 = 4 bytes
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let ins_vert_vec = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ins_vert_vec"),
            size: (max_tets as u64) * 4, // u32 = 4 bytes
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let rev_map_arr = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rev_map_arr"),
            size: (max_tets as u64) * 4, // u32 = 4 bytes
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let act_tet_vec = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("act_tet_vec"),
            size: (max_tets as u64) * 4, // i32 = 4 bytes
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let vote_arr = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vote_arr"),
            size: (max_tets as u64) * 4, // i32 = 4 bytes
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let flip_to_tet = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flip_to_tet"),
            size: (max_tets as u64) * 4, // i32 = 4 bytes
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let flip23_new_slot = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flip23_new_slot"),
            size: (max_tets as u64) * 4, // i32 = 4 bytes
            usage: storage_rw,
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
            flip_arr,
            tet_msg_arr,
            encoded_face_vi_arr,
            tet_to_flip,
            scatter_arr,
            order_arr,
            ins_vert_vec,
            rev_map_arr,
            act_tet_vec,
            vote_arr,
            flip_to_tet,
            flip23_new_slot,
            breadcrumbs,
            thread_debug,
            update_debug,
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

    /// Read compaction counter (from counters[0] after compact_if_negative pass 1).
    pub async fn read_compact_count(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> u32 {
        let counters = self.read_counters(device, queue).await;
        counters.free_count // Using free_count as counter[0]
    }

    /// Write a uniform params buffer (vec4<u32>).
    pub fn create_params_buffer(device: &wgpu::Device, params: [u32; 4]) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Read breadcrumbs (u32 per thread)
    pub async fn read_breadcrumbs(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        num_threads: usize,
    ) -> Vec<u32> {
        self.read_buffer_as(device, queue, &self.breadcrumbs, num_threads)
            .await
    }

    /// Read debug slots for a specific thread (16 vec4<u32>)
    pub async fn read_thread_debug(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        thread_id: usize,
    ) -> Vec<[u32; 4]> {
        let offset = thread_id * 16 * 16; // 16 vec4<u32> per thread
        let size = 16 * 16; // bytes

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("thread_debug_staging"),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.thread_debug, offset as u64, &staging, 0, size as u64);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().unwrap();

        let data = slice.get_mapped_range();
        let raw: Vec<u32> = bytemuck::cast_slice(&data[..]).to_vec();
        drop(data);
        staging.unmap();

        // Convert to array of [u32; 4]
        raw.chunks_exact(4).map(|chunk| [chunk[0], chunk[1], chunk[2], chunk[3]]).collect()
    }

    /// Read update_debug data for the first num_threads threads.
    /// Returns a vector where each element is 4 vec4<u32> values for one thread.
    pub async fn read_update_debug(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        num_threads: usize,
    ) -> Vec<[[u32; 4]; 4]> {
        let size = num_threads * 4 * 16; // 4 vec4<u32> per thread

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("update_debug_staging"),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.update_debug, 0, &staging, 0, size as u64);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().unwrap();

        let data = slice.get_mapped_range();
        let raw: Vec<u32> = bytemuck::cast_slice(&data[..]).to_vec();
        drop(data);
        staging.unmap();

        // Convert to array of [[u32; 4]; 4] (4 vec4s per thread)
        raw.chunks_exact(16).map(|chunk| {
            [
                [chunk[0], chunk[1], chunk[2], chunk[3]],
                [chunk[4], chunk[5], chunk[6], chunk[7]],
                [chunk[8], chunk[9], chunk[10], chunk[11]],
                [chunk[12], chunk[13], chunk[14], chunk[15]],
            ]
        }).collect()
    }
}
