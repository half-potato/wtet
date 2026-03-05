//! Step-by-step debugging of 4-tet allocation in split kernel

use pollster::block_on;
use wgpu::util::DeviceExt;

const MEAN_VERTEX_DEGREE: u32 = 64;

/// Helper to set up minimal GPU environment
async fn setup_gpu() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("test"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            },
        )
        .await
        .unwrap()
}

/// Step 1: Allocate + Write tets (no adjacency, no old tet handling)
#[test]
fn step1_allocate_and_write_tets() {
    block_on(async {
        let (device, queue) = setup_gpu().await;

        // Setup buffers (minimal)
        let num_vertices = 8u32;
        let max_tets = 512u32;

        // Initialize free_arr
        let free_arr_size = (num_vertices * MEAN_VERTEX_DEGREE) as usize;
        let mut free_data = vec![0xFFFFFFFFu32; free_arr_size];
        let mut tet_idx = 1u32;
        for vertex in 0..num_vertices as usize {
            let block_start = vertex * MEAN_VERTEX_DEGREE as usize;
            let block_end = block_start + MEAN_VERTEX_DEGREE as usize;
            for _slot in block_start..block_end {
                if tet_idx < max_tets {
                    free_data[block_start + (tet_idx - 1 - vertex as u32 * MEAN_VERTEX_DEGREE) as usize] = tet_idx;
                    tet_idx += 1;
                } else {
                    break;
                }
            }
        }

        // Simpler init: just fill sequentially
        for vertex in 0..num_vertices as usize {
            let block_start = vertex * MEAN_VERTEX_DEGREE as usize;
            for i in 0..MEAN_VERTEX_DEGREE as usize {
                if vertex * MEAN_VERTEX_DEGREE as usize + i + 1 < max_tets as usize {
                    free_data[block_start + i] = (vertex * MEAN_VERTEX_DEGREE as usize + i + 1) as u32;
                }
            }
        }

        let free_arr = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("free_arr"),
            contents: bytemuck::cast_slice(&free_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let vert_free_data = vec![MEAN_VERTEX_DEGREE; num_vertices as usize];
        let vert_free_arr = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vert_free_arr"),
            contents: bytemuck::cast_slice(&vert_free_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Tets buffer (for writing new tets)
        let tets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tets"),
            size: (max_tets as u64) * 16, // vec4<u32> = 16 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Output: success counter
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("step1"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
@group(0) @binding(0) var<storage, read> free_arr: array<u32>;
@group(0) @binding(1) var<storage, read_write> vert_free_arr: array<u32>;
@group(0) @binding(2) var<storage, read_write> tets: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;

const MEAN_VERTEX_DEGREE: u32 = 64u;

fn get_free_slots_4tet(vertex: u32) -> vec4<u32> {
    let block_top = (vertex + 1u) * MEAN_VERTEX_DEGREE - 1u;
    var slots: vec4<u32>;
    slots.x = free_arr[block_top];
    slots.y = free_arr[block_top - 1u];
    slots.z = free_arr[block_top - 2u];
    slots.w = free_arr[block_top - 3u];
    vert_free_arr[vertex] -= 4u;
    return slots;
}

@compute @workgroup_size(1)
fn test() {
    // Allocate 4 tets for vertex 0
    let p = 4u;  // First real vertex (0-3 are super-tet)
    let slots = get_free_slots_4tet(p);
    let t0 = slots.x;
    let t1 = slots.y;
    let t2 = slots.z;
    let t3 = slots.w;

    // Write 4 new tets (with dummy data)
    tets[t0] = vec4<u32>(p, 1u, 2u, 3u);  // T0 = (P, v1, v2, v3)
    tets[t1] = vec4<u32>(0u, p, 2u, 3u);  // T1 = (v0, P, v2, v3)
    tets[t2] = vec4<u32>(0u, 1u, p, 3u);  // T2 = (v0, v1, P, v3)
    tets[t3] = vec4<u32>(0u, 1u, 2u, p);  // T3 = (v0, v1, v2, P)

    // Success!
    output[0] = 1u;
}
"#
                .into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: free_arr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vert_free_arr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("test"),
            cache: None,
            compilation_options: Default::default(),
        });

        // Execute
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        // Read result
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, 4);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        rx.await.unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: &[u32] = bytemuck::cast_slice(&data);

        println!("Step 1 (Allocate + Write): success = {}", result[0]);
        assert_eq!(result[0], 1, "Should succeed");

        // Also verify tets were written correctly
        let tet_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tet_staging"),
            size: 64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&tets, 0, &tet_staging, 0, 64);
        queue.submit(Some(encoder.finish()));

        let slice = tet_staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        rx.await.unwrap().unwrap();

        let data = slice.get_mapped_range();
        let tet_data: &[[u32; 4]] = bytemuck::cast_slice(&data);

        println!("Tet data written:");
        for (i, tet) in tet_data.iter().take(4).enumerate() {
            println!("  tet[{}] = {:?}", i, tet);
        }
    });
}

/// Step 2: Test using old_tet variable + marking it dead
#[test]
fn step2_with_old_tet_variable() {
    block_on(async {
        let (device, queue) = setup_gpu().await;

        // Same setup as step1...
        let num_vertices = 8u32;
        let max_tets = 512u32;

        let free_arr_size = (num_vertices * MEAN_VERTEX_DEGREE) as usize;
        let mut free_data = vec![0xFFFFFFFFu32; free_arr_size];
        for vertex in 0..num_vertices as usize {
            let block_start = vertex * MEAN_VERTEX_DEGREE as usize;
            for i in 0..MEAN_VERTEX_DEGREE as usize {
                if vertex * MEAN_VERTEX_DEGREE as usize + i + 1 < max_tets as usize {
                    free_data[block_start + i] = (vertex * MEAN_VERTEX_DEGREE as usize + i + 1) as u32;
                }
            }
        }

        let free_arr = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("free_arr"),
            contents: bytemuck::cast_slice(&free_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let vert_free_data = vec![MEAN_VERTEX_DEGREE; num_vertices as usize];
        let vert_free_arr = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vert_free_arr"),
            contents: bytemuck::cast_slice(&vert_free_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let tets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tets"),
            size: (max_tets as u64) * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Add tet_info buffer
        let tet_info = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tet_info"),
            size: (max_tets as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Insert list: simulate one insertion (old_tet=10, point=4)
        let insert_data = vec![[10u32, 4u32]];
        let insert_list = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("insert_list"),
            contents: bytemuck::cast_slice(&insert_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("step2"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
@group(0) @binding(0) var<storage, read> free_arr: array<u32>;
@group(0) @binding(1) var<storage, read_write> vert_free_arr: array<u32>;
@group(0) @binding(2) var<storage, read_write> tets: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> tet_info: array<u32>;
@group(0) @binding(4) var<storage, read> insert_list: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read_write> output: array<u32>;

const MEAN_VERTEX_DEGREE: u32 = 64u;

fn get_free_slots_4tet(vertex: u32) -> vec4<u32> {
    let block_top = (vertex + 1u) * MEAN_VERTEX_DEGREE - 1u;
    var slots: vec4<u32>;
    slots.x = free_arr[block_top];
    slots.y = free_arr[block_top - 1u];
    slots.z = free_arr[block_top - 2u];
    slots.w = free_arr[block_top - 3u];
    vert_free_arr[vertex] -= 4u;
    return slots;
}

@compute @workgroup_size(1)
fn test() {
    // Read from insert_list (like real split kernel)
    let insert = insert_list[0];
    let old_tet = insert.x;  // THIS IS THE KEY DIFFERENCE
    let p = insert.y;

    // Allocate 4 new tets
    let slots = get_free_slots_4tet(p);
    let t0 = slots.x;
    let t1 = slots.y;
    let t2 = slots.z;
    let t3 = slots.w;

    // Mark old tet as dead
    tet_info[old_tet] = 0u;

    // Write 4 new tets
    tets[t0] = vec4<u32>(p, 1u, 2u, 3u);
    tets[t1] = vec4<u32>(0u, p, 2u, 3u);
    tets[t2] = vec4<u32>(0u, 1u, p, 3u);
    tets[t3] = vec4<u32>(0u, 1u, 2u, p);

    // Success!
    output[0] = 1u;
}
"#
                .into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: free_arr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vert_free_arr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tet_info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: insert_list.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("test"),
            cache: None,
            compilation_options: Default::default(),
        });

        // Execute
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        // Read result
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, 4);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        rx.await.unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: &[u32] = bytemuck::cast_slice(&data);

        println!("Step 2 (with old_tet variable): success = {}", result[0]);
        assert_eq!(result[0], 1, "Should succeed");
    });
}
