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
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            flags: wgpu::InstanceFlags::VALIDATION | wgpu::InstanceFlags::DEBUG,
            ..Default::default()
        });
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
                },
                None,
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
    // For 4 points: max_tets = (4+4)*8 = 64
    let max_tets = state.max_tets as usize;
    let tets = pollster::block_on(state.buffers.read_tets(&device, &queue, max_tets));
    let tet_info: Vec<u32> = pollster::block_on(
        state
            .buffers
            .read_buffer_as(&device, &queue, &state.buffers.tet_info, max_tets),
    );

    eprintln!("\n=== TETRAHEDRAL MESH (first 10 tets) ===");
    for i in 0..10.min(max_tets) {
        if tet_info[i] & 1 != 0 {
            eprintln!("Tet {}: {:?}, status: {:#b} (ALIVE)", i, tets[i], tet_info[i]);
        } else {
            eprintln!("Tet {}: {:?}, status: {:#b} (dead)", i, tets[i], tet_info[i]);
        }
    }

    let alive_count = tet_info.iter().filter(|&s| s & 1 != 0).count();
    eprintln!("\nTotal alive tets: {}", alive_count);
    eprintln!("Expected: 4 (the 4 new tets from iteration 1)");

    // Check vert_tet (position-indexed for uninserted vertices)
    eprintln!("\n=== VERT_TET (starting tet for each uninserted position) ===");
    if !state.uninserted.is_empty() {
        let vert_tet: Vec<u32> = pollster::block_on(
            state
                .buffers
                .read_buffer_as(&device, &queue, &state.buffers.vert_tet, state.uninserted.len()),
        );
        for (idx, &tet_idx) in vert_tet.iter().enumerate() {
            let vert_idx = state.uninserted[idx];
            let alive = if (tet_idx as usize) < tet_info.len() && tet_info[tet_idx as usize] & 1 != 0 {
                "ALIVE"
            } else {
                "DEAD"
            };
            eprintln!("Position {} (vertex {}): starts at tet {} ({})", idx, vert_idx, tet_idx, alive);
        }
        eprintln!("\n⚠ If uninserted positions start at DEAD tets, point location will fail!");
    } else {
        eprintln!("All vertices have been inserted!");
    }

    // Read update_uninserted_vert_tet debug data
    let update_debug = pollster::block_on(state.buffers.read_update_debug(&device, &queue, 4));
    eprintln!("\n=== UPDATE_UNINSERTED_VERT_TET DEBUG ===");
    for (tid, data) in update_debug.iter().enumerate().take(4) {
        eprintln!("Thread {}:", tid);
        eprintln!("  Start: idx={}, vert_idx={}, old_tet={}, num_uninserted={}",
            data[0][0], data[0][1], data[0][2], data[0][3]);
        eprintln!("  Result: found_tet={}, new_tet={}, old_status={:#b}",
            data[1][0], data[1][1], data[1][2]);
        eprintln!("  Tet statuses: tet1={:#b}, tet61={:#b}, tet62={:#b}, tet63={:#b}",
            data[2][0], data[2][1], data[2][2], data[2][3]);
        eprintln!("  tet64={:#b}", data[3][0]);
    }

    // List all alive tets and their adjacency
    let tet_opp: Vec<[u32; 4]> = pollster::block_on(
        state
            .buffers
            .read_buffer_as(&device, &queue, &state.buffers.tet_opp, max_tets),
    );

    eprintln!("\n=== ALIVE TETS ===");
    for (i, &status) in tet_info.iter().enumerate().take(max_tets) {
        if status & 1 != 0 {
            eprintln!("Tet {}: {:?}, status: {:#b}", i, tets[i], status);
            eprintln!("  Adjacency: {:?}", tet_opp[i]);
            // Check if any adjacent tets are dead
            for (face, &adj_tet) in tet_opp[i].iter().enumerate() {
                let adj_idx = adj_tet as usize;
                if adj_idx < max_tets && (tet_info[adj_idx] & 1) == 0 {
                    eprintln!("    ⚠ Face {} points to DEAD tet {}", face, adj_tet);
                }
            }
        }
    }

    eprintln!("\n=== SUMMARY ===");
    eprintln!("✓ Vertex layout:");
    eprintln!("  - Vertices 0-3: Real input points (from user input)");
    eprintln!("  - Vertices 4-7: Super-tetrahedron vertices");
    eprintln!("✓ All {} real points should be inserted", points.len());

    let inserted = points.len() - state.uninserted.len();
    if inserted == points.len() {
        eprintln!("✓ SUCCESS: All {} points were inserted!", points.len());
    } else {
        eprintln!("✗ FAILURE: Only {} of {} points were inserted", inserted, points.len());
    }
}
