//! Phase 1: GPU point insertion + flipping.
//!
//! The main loop:
//! 1. Upload uninserted points
//! 2. Locate containing tets (stochastic walk)
//! 3. Vote: each tet picks closest uninserted point
//! 4. Split: insert winning points (1-to-4 split)
//! 5. Flip: check and fix Delaunay violations
//! 6. Remove inserted points from uninserted list
//! 7. Repeat until all points inserted or budget exhausted

use crate::gpu::GpuState;
use crate::types::GDelConfig;

pub async fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    state: &mut GpuState,
    config: &GDelConfig,
) {
    // Step 0: Initialize super-tetrahedron
    {
        let mut encoder = device.create_command_encoder(&Default::default());
        state.dispatch_init(&mut encoder);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    let mut iteration = 0u32;

    while !state.uninserted.is_empty() && iteration < config.max_insert_iterations {
        iteration += 1;
        let num_uninserted = state.uninserted.len() as u32;

        log::debug!(
            "Phase 1 iteration {}: {} points remaining",
            iteration,
            num_uninserted
        );

        // Upload current uninserted list
        state.buffers.upload_uninserted(queue, &state.uninserted);

        // Debug: Check vert_tet at start of iteration
        if iteration == 2 {
            let vert_tet_debug: Vec<u32> = state
                .buffers
                .read_buffer_as(device, queue, &state.buffers.vert_tet, state.uninserted.len())
                .await;
            println!("[DEBUG] Iteration 2 start: vert_tet = {:?}", vert_tet_debug);
        }

        // Expand tetrahedron list to make room for new insertions
        // Port of expandTetraList() call from GpuDelaunay.cu:844
        {
            let mut encoder = device.create_command_encoder(&Default::default());
            state.expand_tetra_list(&mut encoder, queue, num_uninserted);
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);
        }

        // Reset counters for this iteration
        state.reset_inserted_counter(queue);

        // Encode the insertion pipeline
        let mut encoder = device.create_command_encoder(&Default::default());

        // 1. Reset votes (vert_sphere, tet_sphere, tet_vert)
        state.dispatch_reset_votes(&mut encoder);

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // 2. Point location (find containing tet for each uninserted point)
        let mut encoder = device.create_command_encoder(&Default::default());
        state.dispatch_locate(&mut encoder, queue, num_uninserted);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // Debug: Check vert_tet after locate
        if iteration == 2 {
            let vert_tet_after_locate: Vec<u32> = state
                .buffers
                .read_buffer_as(device, queue, &state.buffers.vert_tet, state.uninserted.len())
                .await;
            println!("[DEBUG] Iteration 2 after locate: vert_tet = {:?}", vert_tet_after_locate);
        }

        // 3. Vote (each vertex votes for its tet)
        let mut encoder = device.create_command_encoder(&Default::default());
        state.dispatch_vote(&mut encoder, queue, num_uninserted);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // 4. Pick winners (determine winning vertex per tet and build insert list)
        let mut encoder = device.create_command_encoder(&Default::default());
        state.dispatch_pick_winner(&mut encoder, queue);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // Read back how many points were picked for insertion
        let counters = state.buffers.read_counters(device, queue).await;
        let num_inserted = counters.inserted_count;

        println!("[DEBUG] Iteration {}: num_inserted = {}", iteration, num_inserted);

        if num_inserted == 0 {
            // Debug: Read back vert_tet and tet_vert to see why no winners
            println!("[DEBUG] No winners found! Investigating...");
            println!("[DEBUG] Remaining uninserted: {:?}", state.uninserted);

            log::warn!("No points inserted in iteration {} — breaking", iteration);
            break;
        }

        println!(
            "[DEBUG] Iteration {}: Inserting {} points (uninserted before removal: {})",
            iteration,
            num_inserted,
            num_uninserted
        );

        // 5a. Mark split (for concurrent split detection)
        let mut encoder = device.create_command_encoder(&Default::default());
        state.dispatch_mark_split(&mut encoder, queue, num_inserted);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // 5c. Split points (update vert_tet for vertices whose tets are splitting)
        // Reset scratch counters
        queue.write_buffer(&state.buffers.counters, 16, bytemuck::cast_slice(&[0u32, 0u32, 0u32]));

        let mut encoder = device.create_command_encoder(&Default::default());
        state.dispatch_split_points(&mut encoder, queue, num_uninserted);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // Debug: Check how many vert_tet updates split_points made
        if iteration <= 2 {
            let counters = state.buffers.read_counters(device, queue).await;
            println!("[DEBUG] Iteration {}: split_points: {} threads entered, {} had valid tet, {} updated",
                iteration, counters.scratch[1], counters.scratch[2], counters.scratch[0]);
        }

        // 5d. Split
        let mut encoder = device.create_command_encoder(&Default::default());
        state.dispatch_split(&mut encoder, queue, num_inserted);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // 5e. Split fixup (DISABLED - split.wgsl handles adjacency inline)
        // let mut encoder = device.create_command_encoder(&Default::default());
        // state.dispatch_split_fixup(&mut encoder, queue, num_inserted);
        // queue.submit(Some(encoder.finish()));
        // device.poll(wgpu::Maintain::Wait);

        // Note: We don't update vert_tet here because it will be done at the start of the next iteration

        // 6. Flip checking (optional, iterative) - PROPER 6-STEP CUDA FLOW
        if config.enable_flipping {
            // TODO: Mark special tets before flipping (clears special adjacency markers)
            // state.dispatch_mark_special_tets(&mut encoder, queue);

            // Initialize active tet vector with newly split tets (4 per insertion)
            // TODO: Populate act_tet_vec from split operation
            let mut flip_queue_size = num_inserted * 4;
            let vote_offset = 0u32; // TODO: Implement vote offset management
            let use_alternate = false; // TODO: Implement double buffering if needed

            for flip_iter in 0..config.max_flip_iterations {
                if flip_queue_size == 0 {
                    break;
                }

                // Reset compaction counters
                state.reset_inserted_counter(queue); // Reuse for compaction counters

                let mut encoder = device.create_command_encoder(&Default::default());

                // STEP 1: VOTE - Check Delaunay violations and vote for flips
                state.dispatch_check_delaunay_fast(&mut encoder, queue, flip_queue_size, vote_offset);
                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::Maintain::Wait);

                // STEP 2: VALIDATE - Mark rejected flips
                let mut encoder = device.create_command_encoder(&Default::default());
                state.dispatch_mark_rejected_flips(&mut encoder, queue, flip_queue_size, vote_offset as i32, true);
                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::Maintain::Wait);

                // STEP 3: COMPACT - Remove rejected flips
                let mut encoder = device.create_command_encoder(&Default::default());
                state.dispatch_compact_if_negative(&mut encoder, queue, flip_queue_size);
                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::Maintain::Wait);

                // Read compacted flip count
                let flip_count = state.buffers.read_compact_count(device, queue).await;

                log::debug!(
                    "Flip iteration {}: {} active -> {} valid flips",
                    flip_iter,
                    flip_queue_size,
                    flip_count
                );

                if flip_count == 0 {
                    break;
                }

                // STEP 4: ALLOCATE - Reserve slots for 2-3 flips
                let mut encoder = device.create_command_encoder(&Default::default());
                let inf_idx = state.max_tets; // Infinity vertex at end
                state.dispatch_allocate_flip23_slot(&mut encoder, queue, flip_count, inf_idx, state.max_tets);
                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::Maintain::Wait);

                // STEP 5: FLIP - Perform the flips
                let mut encoder = device.create_command_encoder(&Default::default());
                state.dispatch_flip(&mut encoder, queue, flip_count, use_alternate);
                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::Maintain::Wait);

                // STEP 6: UPDATE - Fix adjacency
                let mut encoder = device.create_command_encoder(&Default::default());
                state.dispatch_update_opp(&mut encoder, queue, 0, flip_count);
                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::Maintain::Wait);

                // Read back how many tets were enqueued for the next round
                let new_count = state.buffers.read_flip_count(device, queue).await;
                flip_queue_size = new_count;
            }

            // CRITICAL: Relocate points after flipping (updates vert_tet for points whose tets flipped)
            let mut encoder = device.create_command_encoder(&Default::default());
            state.dispatch_relocate_points_fast(&mut encoder, queue, num_uninserted);
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);
        }

        // 7. Remove inserted points from uninserted list and compact vert_tet to match.
        // Read back the insert_list to know which vertices were inserted.
        let inserted_verts = read_inserted_verts(device, queue, state, num_inserted as usize).await;
        let inserted_positions: std::collections::HashSet<usize> = inserted_verts
            .iter()
            .filter_map(|v| state.uninserted.iter().position(|u| u == v))
            .collect();

        // Read back current vert_tet
        let old_vert_tet: Vec<u32> = state
            .buffers
            .read_buffer_as(device, queue, &state.buffers.vert_tet, num_uninserted as usize)
            .await;

        // Remove inserted vertices from uninserted
        state
            .uninserted
            .retain(|v| !inserted_verts.contains(v));

        // Compact vert_tet to match new uninserted array
        // Old position -> new position (or None if inserted)
        let mut new_vert_tet = Vec::with_capacity(state.uninserted.len());
        for (old_pos, &tet_idx) in old_vert_tet.iter().enumerate() {
            if !inserted_positions.contains(&old_pos) {
                new_vert_tet.push(tet_idx);
            }
        }

        // Write back compacted vert_tet
        if !new_vert_tet.is_empty() {
            queue.write_buffer(
                &state.buffers.vert_tet,
                0,
                bytemuck::cast_slice(&new_vert_tet),
            );
        }

        println!(
            "[DEBUG] After iteration {}: {} points remaining, vert_tet compacted: {:?}",
            iteration,
            state.uninserted.len(),
            new_vert_tet
        );
    }

    // Final: gather failed vertices
    {
        let mut encoder = device.create_command_encoder(&Default::default());
        state.dispatch_gather(&mut encoder);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    if !state.uninserted.is_empty() {
        println!(
            "[WARNING] Phase 1 complete after {} iterations, {} points FAILED to insert: {:?}",
            iteration,
            state.uninserted.len(),
            state.uninserted
        );
    } else {
        println!(
            "[INFO] Phase 1 complete after {} iterations, all points inserted",
            iteration
        );
    }
}

/// Read back the vertex indices that were inserted this iteration.
async fn read_inserted_verts(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    state: &GpuState,
    count: usize,
) -> Vec<u32> {
    if count == 0 {
        return Vec::new();
    }

    // insert_list is vec2<u32> (tet_idx, position) — pair[1] is position in uninserted array
    let raw: Vec<[u32; 2]> = state
        .buffers
        .read_buffer_as(device, queue, &state.buffers.insert_list, count)
        .await;

    // Convert positions to actual vertex indices
    raw.iter()
        .map(|pair| {
            let position = pair[1] as usize;
            state.uninserted[position]
        })
        .collect()
}
