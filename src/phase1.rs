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
use std::time::Instant;

pub async fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    state: &mut GpuState,
    config: &GDelConfig,
) {
    let phase1_start = Instant::now();

    let mut iteration = 0u32;

    while state.num_uninserted > 0 && iteration < config.max_insert_iterations {
        let iter_start = Instant::now();
        iteration += 1;
        let num_uninserted = state.num_uninserted;

        log::debug!(
            "Phase 1 iteration {}: {} points remaining",
            iteration,
            num_uninserted
        );

        // Note: uninserted buffer is already on GPU and gets compacted in place each iteration

        // Debug: Check vert_tet at start of iteration (DISABLED - too verbose for large datasets)
        // if iteration == 2 {
        //     let num_pos = num_uninserted as usize;
        //     let vert_tet_debug: Vec<u32> = state
        //         .buffers
        //         .read_buffer_as(device, queue, &state.buffers.vert_tet, num_pos)
        //         .await;
        //     println!("[DEBUG] Iteration 2 start: num_uninserted = {}", num_uninserted);
        //     println!("[DEBUG] Iteration 2 start: vert_tet[0..{}] = {:?}", num_pos, &vert_tet_debug[0..num_pos]);
        //
        //     let tet_info_debug: Vec<u32> = state
        //         .buffers
        //         .read_buffer_as(device, queue, &state.buffers.tet_info, 10)
        //         .await;
        //     println!("[DEBUG] Iteration 2 start: tet_info[0..10] = {:?}", tet_info_debug);
        // }

        // Reset counters for this iteration
        state.reset_inserted_counter(queue);

        // CUDA FLOW (GpuDelaunay.cu:788-831):
        // 1. kerVoteForPoint - vote using CURRENT vert_tet (from prev iteration or init)
        // 2. kerPickWinnerPoint - vertex-parallel winner selection
        // 3. kerNegateInsertedVerts - mark winners
        // 4. thrust_copyIf_Insertable - collect winners
        // 5. thrust::gather - get vertex IDs
        // Note: NO point location before voting! Vertices vote for their current tets.

        // BATCHED: Init (iteration 1 only) → Reset votes → Vote → Pick winner → Build insert list
        // OPTIMIZATION: First iteration combines init with vote phase to eliminate one submit/poll cycle
        // This saves ~10-12ms by reducing GPU synchronization overhead
        state.cpu_profiler.begin("1_vote_phase");
        let mut encoder = device.create_command_encoder(&Default::default());

        // Step 0: Initialize super-tetrahedron (first iteration only)
        if iteration == 1 {
            state.dispatch_init(&mut encoder);
        }

        // 1. Reset votes (vert_sphere, tet_sphere, tet_vert)
        // CRITICAL: Must pass num_uninserted to clear correct range
        state.dispatch_reset_votes(&mut encoder, queue, num_uninserted);

        // 2. Vote (each vertex votes for its CURRENT tet - no locate first!)
        state.dispatch_vote(&mut encoder, queue, num_uninserted);

        // 3. Pick winners (vertex-parallel: atomicMin to select lowest index per tet)
        state.dispatch_pick_winner(&mut encoder, queue, num_uninserted);

        // 4. Build insert list (second pass: filter exact winners to prevent duplicates)
        state.dispatch_build_insert_list(&mut encoder, queue, num_uninserted);

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        state.cpu_profiler.end("1_vote_phase");

        // Debug: Read vert_sphere and tet_vote after voting (DISABLED - too verbose for large datasets)
        // if iteration == 2 {
        //     let vert_sphere_debug: Vec<i32> = state
        //         .buffers
        //         .read_buffer_as(device, queue, &state.buffers.vert_sphere, num_uninserted as usize)
        //         .await;
        //     let tet_vote_debug: Vec<i32> = state
        //         .buffers
        //         .read_buffer_as(device, queue, &state.buffers.tet_vote, state.max_tets as usize)
        //         .await;
        //     println!("[DEBUG] After vote: vert_sphere = {:?}", vert_sphere_debug);
        //     println!("[DEBUG] After vote: tet_vote (non-NO_VOTE) = {:?}",
        //         tet_vote_debug.iter().enumerate()
        //             .filter(|(_, &v)| v != i32::MIN)
        //             .collect::<Vec<_>>());
        // }

        // Read back how many points were picked for insertion
        let counters = state.buffers.read_counters(device, queue).await;
        let num_inserted = counters.inserted_count;

        println!("[DEBUG] Iteration {}: num_inserted = {}", iteration, num_inserted);

        if num_inserted == 0 {
            // Debug: Read back vert_tet and tet_vert to see why no winners
            println!("[DEBUG] No winners found! Investigating...");
            println!("[DEBUG] Remaining uninserted count: {}", num_uninserted);

            log::warn!("No points inserted in iteration {} — breaking", iteration);
            break;
        }

        // Read insert_list for expansion
        let insert_list: Vec<[u32; 2]> = state
            .buffers
            .read_buffer_as(device, queue, &state.buffers.insert_list, num_inserted as usize)
            .await;

        if iteration <= 2 {
            println!("[DEBUG] Iteration {}: insert_list = {:?}", iteration, insert_list);
        }

        println!(
            "[DEBUG] Iteration {}: Inserting {} points (uninserted before removal: {})",
            iteration,
            num_inserted,
            num_uninserted
        );

        // 5a. Mark split - REMOVED (redundant)
        // mark_split.wgsl was obsolete - pick_winner_point already populates tet_to_vert
        // See MEMORY.md "Obsolete Shader Cleanup (2026-03-04)"

        // BATCHED: Expand tetra list → Split points → Split
        // This eliminates 2 submit/poll cycles (was 3 separate submissions, now 1)
        // Note: Reset scratch counters inline before dispatching
        queue.write_buffer(&state.buffers.counters, 16, bytemuck::cast_slice(&[0u32, 0u32, 0u32]));

        state.cpu_profiler.begin("2_expand");
        let mut encoder = device.create_command_encoder(&Default::default());

        // Expand tetrahedron list to make room for new insertions
        // Port of expandTetraList() call from GpuDelaunay.cu:844
        // CRITICAL: Expand by num_inserted (winners this iteration), NOT num_uninserted (all remaining)!
        // CUDA: expandTetraList( &realInsVertVec, ... ) where realInsVertVec.size() == _insNum
        state.expand_tetra_list(&mut encoder, queue, num_inserted, &insert_list);

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        state.cpu_profiler.end("2_expand");

        // 5c. Split points (update vert_tet for vertices whose tets are splitting)
        state.cpu_profiler.begin("3_split_points");
        let mut encoder = device.create_command_encoder(&Default::default());
        state.dispatch_split_points(&mut encoder, queue, num_uninserted);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        state.cpu_profiler.end("3_split_points");

        // 5d. Split
        state.cpu_profiler.begin("4_split");
        println!("[DEBUG] Dispatching split with num_inserted = {}", num_inserted);
        let mut encoder = device.create_command_encoder(&Default::default());
        state.dispatch_split(&mut encoder, queue, num_inserted);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        println!("[DEBUG] Batched operations complete");
        state.cpu_profiler.end("4_split");

        // 5e. Split fixup (DISABLED - split.wgsl handles adjacency inline)
        // let mut encoder = device.create_command_encoder(&Default::default());
        // state.dispatch_split_fixup(&mut encoder, queue, num_inserted);
        // queue.submit(Some(encoder.finish()));
        // device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        // Note: We don't update vert_tet here because it will be done at the start of the next iteration

        // 6. Flip checking (optional, iterative) - TWO-PHASE FLIPPING
        if config.enable_flipping {
            // Collect ALL alive tets for flipping. CUDA's check_delaunay runs over
            // all active tets, not just changed ones.
            let tet_num_for_collect = state.current_tet_num;
            let mut flip_queue_size = state
                .dispatch_collect_active_tets(device, queue, tet_num_for_collect)
                .await;
            eprintln!("[FLIP] Collected {} active tets for flipping", flip_queue_size);

            let mut use_alternate = false;

            // Track flip batch boundaries for relocateAll
            let mut org_flip_num: Vec<u32> = Vec::new();
            let mut total_flips = 0u32;

            // ========== PHASE 1: FAST FLIPPING (self-contained flip shader) ==========
            // The flip shader (flip_check) is self-contained: it does its own insphere/orient3d
            // checks, CAS locking, allocation, adjacency updates, and enqueues new tets.
            // We bypass check_delaunay/mark_rejected/allocate/update_opp — those were designed
            // for CUDA's kerFlip which uses pre-computed flip info, but our flip.wgsl doesn't.
            state.cpu_profiler.begin("5_flip_fast");
            eprintln!("[FLIP] Starting fast flipping phase with {} initial tets", flip_queue_size);

            // Copy act_tet_vec → flip_queue to seed the flip shader's input
            // (act_tet_vec is i32, flip_queue is u32 — bit-identical for positive values)
            {
                let mut encoder = device.create_command_encoder(&Default::default());
                encoder.copy_buffer_to_buffer(
                    &state.buffers.act_tet_vec, 0,
                    &state.buffers.flip_queue, 0,
                    (flip_queue_size as u64) * 4,
                );
                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            }

            for flip_iter in 0..config.max_flip_iterations {
                if flip_queue_size == 0 {
                    break;
                }

                // Reset flip_count[0] (next queue size) and flip_count[1] (metadata flip count)
                state.reset_flip_count(queue);

                // Three-phase flip: vote → check+execute → update_opp
                let mut encoder = device.create_command_encoder(&Default::default());
                let inf_idx = state.num_points + 4;

                // Pass 1: Reset tet_vote to INT_MAX
                state.dispatch_reset_flip_votes(&mut encoder, queue);
                // Pass 2: Vote (atomicMin on all involved tets)
                state.dispatch_flip_vote(&mut encoder, queue, flip_queue_size, inf_idx, use_alternate, false, total_flips);
                // Pass 3: Execute flips + write metadata (NO adjacency writes)
                state.dispatch_flip(&mut encoder, queue, flip_queue_size, inf_idx, use_alternate, total_flips);

                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

                // Read counters: (next_queue_size, metadata_flip_count)
                let (new_count, flip_num) = state.buffers.read_flip_count(device, queue).await;

                // Pass 4: update_opp — fix all adjacency using flip metadata
                if flip_num > 0 {
                    let mut encoder = device.create_command_encoder(&Default::default());
                    state.dispatch_update_opp(&mut encoder, queue, total_flips, flip_num);
                    queue.submit(Some(encoder.finish()));
                    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
                }

                eprintln!(
                    "[FLIP] Iteration {}: input={}, flipped_to={}, flips={}",
                    flip_iter + 1, flip_queue_size, new_count, flip_num
                );

                // Track org_flip_num for tet_msg_arr staleness
                org_flip_num.push(total_flips);
                total_flips += flip_num;

                if new_count == 0 {
                    break;
                }

                // Swap flip_queue ↔ flip_queue_next for next iteration
                use_alternate = !use_alternate;
                flip_queue_size = new_count;

                // Safety check for flip array overflow
                if total_flips > state.max_flips {
                    eprintln!("[WARNING] Flip array overflow: {} > {} capacity",
                             total_flips, state.max_flips);
                    break;
                }
            }

            eprintln!("[FLIP] Fast phase complete: {} total flipped tets", total_flips);
            state.cpu_profiler.end("5_flip_fast");

            // ========== PHASE 2: Re-check ALL alive tets (multiple rounds) ==========
            // After fast phase converges, some violations may exist between tets
            // that were not in the queue (created in earlier iterations). CUDA's
            // check_delaunay examines all active tets, not just changed ones.
            // We loop: collect ALL alive → flip → repeat until no new flips.
            {
                state.cpu_profiler.begin("6_reflip_all");
                let tet_num = state.current_tet_num;

                for global_round in 0..5u32 {
                    let all_alive_count = state
                        .dispatch_collect_all_alive_tets(device, queue, tet_num)
                        .await;

                    if all_alive_count == 0 { break; }

                    let mut encoder = device.create_command_encoder(&Default::default());
                    encoder.copy_buffer_to_buffer(
                        &state.buffers.act_tet_vec, 0,
                        if use_alternate { &state.buffers.flip_queue_next } else { &state.buffers.flip_queue }, 0,
                        (all_alive_count as u64) * 4,
                    );
                    queue.submit(Some(encoder.finish()));
                    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

                    let mut round_flips = 0u32;
                    flip_queue_size = all_alive_count;
                    for flip_iter in 0..config.max_flip_iterations {
                        if flip_queue_size == 0 { break; }

                        state.reset_flip_count(queue);
                        let mut encoder = device.create_command_encoder(&Default::default());
                        let inf_idx = state.num_points + 4;

                        // Three-phase flip with exact predicates
                        state.dispatch_reset_flip_votes(&mut encoder, queue);
                        state.dispatch_flip_vote(&mut encoder, queue, flip_queue_size, inf_idx, use_alternate, true, total_flips);
                        state.dispatch_flip_exact(&mut encoder, queue, flip_queue_size, inf_idx, use_alternate, total_flips);

                        queue.submit(Some(encoder.finish()));
                        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

                        let (new_count, flip_num) = state.buffers.read_flip_count(device, queue).await;

                        // update_opp — fix adjacency using flip metadata
                        if flip_num > 0 {
                            let mut encoder = device.create_command_encoder(&Default::default());
                            state.dispatch_update_opp(&mut encoder, queue, total_flips, flip_num);
                            queue.submit(Some(encoder.finish()));
                            device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
                        }

                        eprintln!(
                            "[FLIP-ALL R{}] Iteration {}: input={}, flipped_to={}, flips={}",
                            global_round + 1, flip_iter + 1, flip_queue_size, new_count, flip_num
                        );

                        round_flips += flip_num;
                        total_flips += flip_num;

                        if new_count == 0 { break; }
                        use_alternate = !use_alternate;
                        flip_queue_size = new_count;

                        if total_flips > state.max_flips { break; }
                    }

                    if round_flips == 0 { break; }
                    eprintln!("[FLIP-ALL R{}] Found {} new flips, re-collecting...", global_round + 1, round_flips);
                }

                eprintln!("[FLIP] All-tets phase complete: {} total flipped tets", total_flips);
                state.cpu_profiler.end("6_reflip_all");
            }

            // After flipping, some vert_tet entries may point to dead tets
            // (destroyed by 3-2 flips). Repair them by walking adjacency.
            // This replaces the full relocateAll which requires flip trace arrays
            // that the self-contained flip shader doesn't populate.
            if total_flips > 0 {
                state.cpu_profiler.begin("8_repair_vert_tet");
                let mut encoder = device.create_command_encoder(&Default::default());
                state.dispatch_update_uninserted_vert_tet(&mut encoder, queue, num_uninserted);
                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
                state.cpu_profiler.end("8_repair_vert_tet");
            }
        }

        // 7. Remove inserted points from uninserted list and compact vert_tet to match.
        // GPU-accelerated compaction (FLAW #2 fix)
        state.cpu_profiler.begin("9_compact");
        let new_count_from_compact;  // Declare outside block
        {
            // Adaptive compaction: choose fastest algorithm based on actual work
            // ATOMIC PATH: O(N×M) - fast when work is low (N×M < 10M operations)
            // PREFIX SUM PATH: O(N+M) - better when work is high (higher overhead but scales linearly)
            // Old threshold: num_uninserted < 100_000 (ignored M, causing bad choices)
            // New threshold: N×M < 10_000_000 (considers actual operation count)
            let use_atomic = (num_uninserted as u64 * num_inserted as u64) < 10_000_000;

            let (mut encoder, new_count) = if use_atomic {
                // FAST PATH: Atomic-based compaction (2 passes)
                eprintln!("[COMPACT] Using atomic compaction for {} uninserted", num_uninserted);
                state.dispatch_compact_vertex_arrays_atomic(device, queue, num_uninserted, num_inserted).await
            } else {
                // SLOW PATH: GPU prefix sum compaction (7 passes, optimized)
                eprintln!("[COMPACT] Using prefix sum compaction for {} uninserted", num_uninserted);
                state.dispatch_compact_vertex_arrays(device, queue, num_uninserted, num_inserted).await
            };
            new_count_from_compact = new_count;  // Assign to outer variable

            // Copy compacted results from temp buffers back to main buffers
            encoder.copy_buffer_to_buffer(
                &state.buffers.uninserted_temp,
                0,
                &state.buffers.uninserted,
                0,
                (num_uninserted as u64) * 4,
            );
            encoder.copy_buffer_to_buffer(
                &state.buffers.vert_tet_temp,
                0,
                &state.buffers.vert_tet,
                0,
                (num_uninserted as u64) * 4,
            );

            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        }

        // Use new_count from compaction (no need to read counters anymore)
        let new_count = new_count_from_compact;
        println!(
            "[DEBUG_COMPACT] Iteration {}: num_uninserted={}, num_inserted={}, new_count={}",
            iteration, num_uninserted, num_inserted, new_count
        );

        // Update CPU state (count only, array stays on GPU)
        state.num_uninserted = new_count;

        state.cpu_profiler.end("9_compact");

        let iter_duration = iter_start.elapsed();
        eprintln!("[TIMING] Iteration {} completed in {:.3} ms", iteration, iter_duration.as_secs_f64() * 1000.0);

        println!(
            "[DEBUG] After iteration {}: {} points remaining (GPU compacted)",
            iteration,
            new_count
        );
    }

    // Final: gather failed vertices
    state.cpu_profiler.begin("10_gather");
    {
        let mut encoder = device.create_command_encoder(&Default::default());
        state.dispatch_gather(&mut encoder);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }
    state.cpu_profiler.end("10_gather");

    if state.num_uninserted > 0 {
        println!(
            "[WARNING] Phase 1 complete after {} iterations, {} points FAILED to insert",
            iteration,
            state.num_uninserted
        );
    } else {
        println!(
            "[INFO] Phase 1 complete after {} iterations, all points inserted",
            iteration
        );
    }

    // Print profiling summary
    let phase1_duration = phase1_start.elapsed();
    eprintln!("\n[TIMING] Phase 1 total: {:.2} seconds", phase1_duration.as_secs_f64());
    state.cpu_profiler.print_summary();

    // Collect GPU profiling data if enabled
    if let Some(ref mut gpu_profiler) = state.gpu_profiler {
        gpu_profiler.collect(device, queue, state.timestamp_period).await;
        gpu_profiler.print_summary();
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

    // insert_list is vec2<u32> (tet_idx, vert_idx) — pair[1] is already the vertex ID
    let raw: Vec<[u32; 2]> = state
        .buffers
        .read_buffer_as(device, queue, &state.buffers.insert_list, count)
        .await;

    // Extract vertex IDs directly (no conversion needed)
    raw.iter().map(|pair| pair[1]).collect()
}
