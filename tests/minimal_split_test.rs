//! Minimal reproduction case for split shader not executing.
//!
//! This test creates the absolute minimum setup needed to run split_tetra
//! and verifies if the shader executes at all.

use wgpu::util::DeviceExt;

#[test]
fn test_minimal_split_execution() {
    // 1. Setup GPU
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::VALIDATION | wgpu::InstanceFlags::DEBUG,
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("Failed to find adapter");

    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffers_per_shader_stage = 10;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("test"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: Default::default(),
        },
        None,
    ))
    .expect("Failed to create device");

    eprintln!("[TEST] GPU device created");

    // 2. Create minimal buffers
    let storage_rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

    // Tets: 10 tets * 4 vertices * 4 bytes
    let tets = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tets"),
        size: 10 * 16,
        usage: storage_rw,
        mapped_at_creation: false,
    });

    // tet_opp: 10 tets * 4 faces * 4 bytes
    let tet_opp = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tet_opp"),
        size: 10 * 4 * 4,
        usage: storage_rw,
        mapped_at_creation: false,
    });

    // tet_info: 10 tets * 4 bytes (initialized to 0)
    let tet_info = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tet_info"),
        contents: &vec![0u8; 10 * 4],
        usage: storage_rw,
    });

    // vert_tet: 8 vertices * 4 bytes
    let vert_tet = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vert_tet"),
        size: 8 * 4,
        usage: storage_rw,
        mapped_at_creation: false,
    });

    // insert_list: 1 insertion (tet_idx=0, vertex_idx=1)
    let insert_list = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("insert_list"),
        contents: bytemuck::cast_slice(&[0u32, 1u32]), // (tet=0, vertex=1)
        usage: storage_rw,
    });

    // free_arr: 80 slots (10 vertices * 8)
    let free_arr = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("free_arr"),
        contents: &vec![0u8; 80 * 4],
        usage: storage_rw,
    });

    // vert_free_arr: 8 vertices
    let vert_free_arr = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vert_free_arr"),
        contents: bytemuck::cast_slice(&[8u32; 8]), // Each vertex has 8 free slots
        usage: storage_rw,
    });

    // counters: 12 * 4 bytes (8 counters + 4 scratch)
    let counters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("counters"),
        contents: bytemuck::cast_slice(&[0u32; 12]),
        usage: storage_rw,
    });

    // flip_queue: 40 tets
    let flip_queue = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flip_queue"),
        size: 40 * 4,
        usage: storage_rw,
        mapped_at_creation: false,
    });

    // tet_to_vert: 10 tets
    let tet_to_vert = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tet_to_vert"),
        contents: bytemuck::cast_slice(&[0xFFFFFFFFu32; 10]),
        usage: storage_rw,
    });

    // params: (num_insertions=1, inf_idx=7, current_tet_num=9, 0)
    let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::cast_slice(&[1u32, 7u32, 9u32, 0u32]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    eprintln!("[TEST] Buffers created");

    // 3. Load split shader
    let split_shader_src = include_str!("../src/shaders/split.wgsl");
    let split_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("split.wgsl"),
        source: wgpu::ShaderSource::Wgsl(split_shader_src.into()),
    });

    eprintln!("[TEST] Shader module created");

    // 4. Create bind group layout
    let storage_ro = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };

    let storage_rw = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };

    let uniform = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("split_bgl"),
        entries: &[
            storage_rw(0),  // tets
            storage_rw(1),  // tet_opp
            storage_rw(2),  // tet_info
            storage_rw(3),  // vert_tet
            storage_ro(4),  // insert_list
            storage_rw(5),  // free_arr
            storage_rw(6),  // vert_free_arr
            storage_rw(7),  // counters
            storage_rw(8),  // flip_queue
            storage_rw(9),  // tet_to_vert
            uniform(10),    // params
        ],
    });

    eprintln!("[TEST] Bind group layout created");

    // 5. Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("split_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: tets.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: tet_opp.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: tet_info.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: vert_tet.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: insert_list.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: free_arr.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: vert_free_arr.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: counters.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: flip_queue.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: tet_to_vert.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: params.as_entire_binding() },
        ],
    });

    eprintln!("[TEST] Bind group created");

    // 6. Create pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("split_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("split"),
        layout: Some(&pipeline_layout),
        module: &split_shader,
        entry_point: Some("split_tetra"),
        compilation_options: Default::default(),
        cache: None,
    });

    eprintln!("[TEST] Pipeline created successfully");

    // 7. Dispatch split
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("split"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1); // num_insertions=1, workgroup_size=64
    }

    eprintln!("[TEST] Dispatching split...");
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);
    eprintln!("[TEST] Split dispatched and completed");

    // 8. Read back tet_info to verify split ran
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: 10 * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&tet_info, 0, &staging_buffer, 0, 10 * 4);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    pollster::block_on(rx).unwrap().unwrap();

    let data = buffer_slice.get_mapped_range();
    let tet_info_result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    eprintln!("[TEST] tet_info after split: {:?}", &tet_info_result[0..10]);

    // Check if split set TET_ALIVE flag (bit 0)
    let tet_0_alive = (tet_info_result[0] & 1) != 0;
    let tet_1_alive = (tet_info_result[1] & 1) != 0;
    let tet_2_alive = (tet_info_result[2] & 1) != 0;
    let tet_3_alive = (tet_info_result[3] & 1) != 0;
    let tet_4_alive = (tet_info_result[4] & 1) != 0;

    eprintln!("[TEST] Tet 0 alive: {}", tet_0_alive);
    eprintln!("[TEST] Tet 1 alive: {}", tet_1_alive);
    eprintln!("[TEST] Tet 2 alive: {}", tet_2_alive);
    eprintln!("[TEST] Tet 3 alive: {}", tet_3_alive);
    eprintln!("[TEST] Tet 4 alive: {}", tet_4_alive);

    // Expected: tet 0 should be dead (0), tets 1-4 should be alive (TET_ALIVE=1)
    // But if split didn't run, all will be 0
    if !tet_1_alive && !tet_2_alive && !tet_3_alive && !tet_4_alive {
        panic!("Split shader did NOT execute - all tet_info values are 0!");
    }

    assert!(!tet_0_alive, "Tet 0 should be dead after split");
    assert!(tet_1_alive, "Tet 1 should be alive after split");
    assert!(tet_2_alive, "Tet 2 should be alive after split");
    assert!(tet_3_alive, "Tet 3 should be alive after split");
    assert!(tet_4_alive, "Tet 4 should be alive after split");

    eprintln!("[TEST] ✅ Split shader executed successfully!");
}
