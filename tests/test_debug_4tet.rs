/// Test using breadcrumb trail and debug slots to find the 4-tet allocation bug
use gdel3d_wgpu::*;

#[test]
fn test_debug_4tet() {
    let config = types::GDelConfig::default();
    // Use normalized points directly
    let points = vec![
        [0.1, 0.1, 0.1],
        [0.9, 0.1, 0.1],
        [0.5, 0.9, 0.1],
        [0.5, 0.5, 0.9],
    ];

    let (device, queue) = pollster::block_on(async {
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
                    required_limits: required_limits(),
                    memory_hints: wgpu::MemoryHints::default(),
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    trace: wgpu::Trace::Off,
                },
            )
            .await
            .unwrap()
    });

    let mut state =
        pollster::block_on(gdel3d_wgpu::gpu::GpuState::new(&device, &queue, &points, &config));

    // Enable logging to see what phase1 is doing
    std::env::set_var("RUST_LOG", "gdel3d_wgpu=debug");
    env_logger::try_init().ok();

    // Run phase1 (this will attempt splits)
    pollster::block_on(gdel3d_wgpu::phase1::run(&device, &queue, &mut state, &config));

    // Read debug data
    let breadcrumbs = pollster::block_on(state.buffers.read_breadcrumbs(&device, &queue, 10));

    eprintln!("\n=== BREADCRUMB TRAIL ===");
    for (tid, crumb) in breadcrumbs.iter().enumerate().take(10) {
        let status = match *crumb {
            0 => "NOT_STARTED".to_string(),
            1 => "START".to_string(),
            2 => "READ_INSERT".to_string(),
            3 => "AFTER_ALLOC".to_string(),
            4 => "BEFORE_WRITE".to_string(),
            5 => "AFTER_WRITE_T0".to_string(),
            6 => "AFTER_WRITE_ALL".to_string(),
            7 => "AFTER_MARK_DEAD".to_string(),
            8 => "AFTER_ADJACENCY".to_string(),
            99 => "COMPLETE ✓".to_string(),
            0xDEAD => "ALLOC_FAILED".to_string(),
            other => format!("UNKNOWN({})", other),
        };
        eprintln!("Thread {}: {}", tid, status);
    }

    // Read debug slots for first few threads
    eprintln!("\n=== DEBUG SLOTS ===");
    for tid in 0..4 {
        let slots = pollster::block_on(state.buffers.read_thread_debug(&device, &queue, tid));
        eprintln!("Thread {}:", tid);
        eprintln!("  Slot 0 (old_tet, p): {:?}", slots[0]);
        eprintln!("  Slot 1 (allocated tets): {:?}", slots[1]);
        eprintln!("  Slot 2 (final tets): {:?}", slots[2]);
    }

    // Read counters for context
    let counters = pollster::block_on(state.buffers.read_counters(&device, &queue));
    eprintln!("\n=== COUNTERS ===");
    eprintln!("Active tets: {}", counters.active_count);
    eprintln!("Inserted count: {}", counters.inserted_count);
    eprintln!("Uninserted remaining: {}", state.uninserted.len());

    // Analysis
    eprintln!("\n=== ANALYSIS ===");
    let incomplete: Vec<_> = breadcrumbs
        .iter()
        .enumerate()
        .filter(|(_, c)| **c != 0 && **c != 99)
        .collect();

    if !incomplete.is_empty() {
        eprintln!("⚠ {} threads did not complete:", incomplete.len());
        for (tid, crumb) in incomplete {
            eprintln!("  Thread {} stopped at crumb {}", tid, crumb);
        }
    }

    let not_started: usize = breadcrumbs.iter().filter(|c| **c == 0).count();
    eprintln!("Threads that never started: {}", not_started);

    let completed: usize = breadcrumbs.iter().filter(|c| **c == 99).count();
    eprintln!("Threads that completed: {}", completed);

    // Read actual tets and their status to verify
    let tets = pollster::block_on(state.buffers.read_tets(&device, &queue, 100));
    let tet_info: Vec<u32> = pollster::block_on(
        state
            .buffers
            .read_buffer_as(&device, &queue, &state.buffers.tet_info, 100),
    );

    eprintln!("\n=== TETRAHEDRAL MESH ===");
    eprintln!("Tet 0 (super-tet): {:?}, status: {:#b}", tets[0], tet_info[0]);
    eprintln!("Tet 64 (new): {:?}, status: {:#b}", tets[64], tet_info[64]);
    eprintln!("Tet 63 (new): {:?}, status: {:#b}", tets[63], tet_info[63]);
    eprintln!("Tet 62 (new): {:?}, status: {:#b}", tets[62], tet_info[62]);
    eprintln!("Tet 61 (new): {:?}, status: {:#b}", tets[61], tet_info[61]);

    let alive_count = tet_info.iter().filter(|&s| s & 1 != 0).count();
    eprintln!("\nTotal alive tets: {}", alive_count);
    eprintln!("Expected: 4 (the 4 new tets from iteration 1)");

    // Check vert_tet (where each point thinks it should start searching)
    let vert_tet: Vec<u32> = pollster::block_on(
        state
            .buffers
            .read_buffer_as(&device, &queue, &state.buffers.vert_tet, 8),
    );
    eprintln!("\n=== VERT_TET (starting tet for each vertex) ===");
    for i in 0..8 {
        let alive = if i < tet_info.len() && tet_info[vert_tet[i] as usize] & 1 != 0 {
            "ALIVE"
        } else {
            "DEAD"
        };
        eprintln!("Vertex {}: starts at tet {} ({})", i, vert_tet[i], alive);
    }

    eprintln!("\n⚠ If uninserted points (1,2,3) start at DEAD tets, point location will fail!");

    // List all alive tets
    eprintln!("\n=== ALIVE TETS ===");
    for (i, &status) in tet_info.iter().enumerate().take(100) {
        if status & 1 != 0 {
            eprintln!("Tet {}: {:?}, status: {:#b}", i, tets[i], status);
        }
    }

    eprintln!("\n⚠ CRITICAL: Thread 0 inserted point p={}", 0);
    eprintln!("  Expected: p should be 4,5,6,or 7 (first real point)");
    eprintln!("  Actual: p=0 is a super-tet vertex!");
    eprintln!("  This suggests insert_list was populated incorrectly.");
}
