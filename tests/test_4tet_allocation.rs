//! Minimal isolated test for 4-tet allocation bug
//!
//! This test ONLY exercises the allocation logic, without any of the
//! Delaunay triangulation complexity. It should help identify exactly
//! where 4-tet allocation breaks.

use pollster::block_on;
use wgpu::util::DeviceExt;

const MEAN_VERTEX_DEGREE: u32 = 64;

/// Minimal test: Just allocate 4 tets and verify we got valid indices
#[test]
fn test_minimal_4tet_allocation() {
    block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("test"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .unwrap();

        // Setup: Simple case - 8 vertices, 512 tets
        let num_vertices = 8u32;
        let max_tets = 512u32;

        // Initialize free_arr (same as production code)
        let free_arr_size = (num_vertices * MEAN_VERTEX_DEGREE) as usize;
        let mut free_data = vec![0xFFFFFFFFu32; free_arr_size];
        let mut tet_idx = 1u32;
        for vertex in 0..num_vertices as usize {
            let block_start = vertex * MEAN_VERTEX_DEGREE as usize;
            let block_end = block_start + MEAN_VERTEX_DEGREE as usize;
            for slot in block_start..block_end {
                if tet_idx < max_tets {
                    free_data[slot] = tet_idx;
                    tet_idx += 1;
                } else {
                    break;
                }
            }
        }

        // Initialize vert_free_arr (with bug fix!)
        let mut vert_free_data = vec![0u32; num_vertices as usize];
        let mut tet_idx = 1u32;
        for vertex in 0..num_vertices as usize {
            let block_start = vertex * MEAN_VERTEX_DEGREE as usize;
            let block_end = block_start + MEAN_VERTEX_DEGREE as usize;
            let mut count = 0u32;
            for slot in block_start..block_end {
                if tet_idx < max_tets {
                    tet_idx += 1;
                    count += 1;
                } else {
                    break;
                }
            }
            vert_free_data[vertex] = count;
        }

        // Create buffers
        let free_arr = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("free_arr"),
            contents: bytemuck::cast_slice(&free_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let vert_free_arr = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vert_free_arr"),
            contents: bytemuck::cast_slice(&vert_free_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Output buffer for allocated tet indices
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: 16, // 4 × u32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Shader: Just call get_free_slots_4tet and write results
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test_shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
@group(0) @binding(0) var<storage, read> free_arr: array<u32>;
@group(0) @binding(1) var<storage, read_write> vert_free_arr: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

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
fn test_allocation() {
    // Test vertex 0 allocation
    let slots = get_free_slots_4tet(0u);

    output[0] = slots.x;
    output[1] = slots.y;
    output[2] = slots.z;
    output[3] = slots.w;
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
                    resource: output.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("test_allocation"),
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
        device.poll(wgpu::Maintain::Wait);

        // Read back results
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, 16);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().unwrap();

        let data = slice.get_mapped_range();
        let results: &[u32] = bytemuck::cast_slice(&data);

        println!("Allocated tets: {:?}", results);
        println!("Expected: [64, 63, 62, 61]");

        // Verify
        assert_eq!(results[0], 64, "t0 should be 64");
        assert_eq!(results[1], 63, "t1 should be 63");
        assert_eq!(results[2], 62, "t2 should be 62");
        assert_eq!(results[3], 61, "t3 should be 61");

        // Also check vert_free_arr was decremented
        let vert_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vert_staging"),
            size: 32, // 8 × u32
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&vert_free_arr, 0, &vert_staging, 0, 32);
        queue.submit(Some(encoder.finish()));

        let slice = vert_staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().unwrap();

        let data = slice.get_mapped_range();
        let vert_counts: &[u32] = bytemuck::cast_slice(&data);

        println!("vert_free_arr: {:?}", vert_counts);
        println!("Expected vertex 0: {} (was 64, now 60)", vert_counts[0]);

        assert_eq!(vert_counts[0], 60, "Vertex 0 should have 60 free tets after allocating 4");
    });
}

/// Test: Does using t0 vs old_tet matter?
#[test]
fn test_variable_naming() {
    block_on(async {
        // Test 1: Use t0 name (allocated from free_arr)
        // Test 2: Use old_tet name (original from insert_list)
        // Compare results - does naming matter?

        // TODO: Implement this to test the hypothesis
    });
}

/// Test: Full split operation with 4-tet allocation
#[test]
fn test_minimal_split_with_4tet() {
    // TODO: Test the FULL split kernel logic:
    // - Read original tet
    // - Allocate 4 new tets (PASSES - we know this works!)
    // - Write 4 new tets
    // - Update adjacency
    // - Mark old tet dead
    // - Update counters
    //
    // This should reveal which STEP after allocation fails
}
