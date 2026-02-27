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

        // Reset counters for this iteration
        state.reset_inserted_counter(queue);

        // Encode the insertion pipeline
        let mut encoder = device.create_command_encoder(&Default::default());

        // 1. Reset votes
        state.dispatch_reset_votes(&mut encoder);

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // 2. Point location (find containing tet for each uninserted point)
        let mut encoder = device.create_command_encoder(&Default::default());
        state.dispatch_locate(&mut encoder, queue, num_uninserted);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // 3. Vote
        let mut encoder = device.create_command_encoder(&Default::default());
        state.dispatch_vote(&mut encoder, queue, num_uninserted);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // 4. Pick winners
        let mut encoder = device.create_command_encoder(&Default::default());
        state.dispatch_pick_winner(&mut encoder, queue);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // Read back how many points were picked for insertion
        let counters = state.buffers.read_counters(device, queue).await;
        let num_inserted = counters.inserted_count;

        if num_inserted == 0 {
            log::warn!("No points inserted in iteration {} — breaking", iteration);
            break;
        }

        log::debug!("Inserting {} points", num_inserted);

        // 5a. Mark split (for concurrent split detection)
        let mut encoder = device.create_command_encoder(&Default::default());
        state.dispatch_mark_split(&mut encoder, queue, num_inserted);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // 5b. Split
        let mut encoder = device.create_command_encoder(&Default::default());
        state.dispatch_split(&mut encoder, queue, num_inserted);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // 6. Flip checking (optional, iterative)
        if config.enable_flipping {
            let mut flip_queue_size = num_inserted * 4; // 4 tets per insertion
            let mut use_alternate = false;

            for flip_iter in 0..config.max_flip_iterations {
                if flip_queue_size == 0 {
                    break;
                }

                state.reset_flip_count(queue);

                let mut encoder = device.create_command_encoder(&Default::default());
                state.dispatch_flip(&mut encoder, queue, flip_queue_size, use_alternate);
                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::Maintain::Wait);

                // Read back how many tets were enqueued for the next round
                let new_count = state.buffers.read_flip_count(device, queue).await;
                log::debug!(
                    "Flip iteration {}: {} -> {} tets",
                    flip_iter,
                    flip_queue_size,
                    new_count
                );

                flip_queue_size = new_count;
                use_alternate = !use_alternate;
            }
        }

        // 7. Remove inserted points from uninserted list.
        // Read back the insert_list to know which vertices were inserted.
        let inserted_verts = read_inserted_verts(device, queue, state, num_inserted as usize).await;
        let inserted_set: std::collections::HashSet<u32> = inserted_verts.into_iter().collect();

        state
            .uninserted
            .retain(|v| !inserted_set.contains(v));

        log::debug!(
            "After iteration {}: {} points remaining",
            iteration,
            state.uninserted.len()
        );
    }

    // Final: gather failed vertices
    {
        let mut encoder = device.create_command_encoder(&Default::default());
        state.dispatch_gather(&mut encoder);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    log::info!(
        "Phase 1 complete after {} iterations, {} points remaining",
        iteration,
        state.uninserted.len()
    );
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

    // insert_list is vec2<u32> (tet_idx, vert_idx) — we want the vert_idx
    let raw: Vec<[u32; 2]> = state
        .buffers
        .read_buffer_as(device, queue, &state.buffers.insert_list, count)
        .await;

    raw.iter().map(|pair| pair[1]).collect()
}
