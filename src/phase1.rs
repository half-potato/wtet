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
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        // Debug: Check if tet 0 is alive after init
        let tet_info_debug: Vec<u32> = state
            .buffers
            .read_buffer_as(device, queue, &state.buffers.tet_info, 1)
            .await;
        println!("[DEBUG] After init: tet_info[0] = {}", tet_info_debug[0]);
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
            // Check vert_tet for uninserted vertices (by POSITION, not vertex ID!)
            let uninserted_verts: Vec<u32> = state.uninserted.clone();
            let num_pos = uninserted_verts.len();
            let vert_tet_debug: Vec<u32> = state
                .buffers
                .read_buffer_as(device, queue, &state.buffers.vert_tet, num_pos)
                .await;
            println!("[DEBUG] Iteration 2 start: uninserted verts = {:?}", uninserted_verts);
            println!("[DEBUG] Iteration 2 start: vert_tet[0..{}] = {:?}", num_pos, &vert_tet_debug[0..num_pos]);

            // Check if those tets are alive
            let tet_info_debug: Vec<u32> = state
                .buffers
                .read_buffer_as(device, queue, &state.buffers.tet_info, 10)
                .await;
            println!("[DEBUG] Iteration 2 start: tet_info[0..10] = {:?}", tet_info_debug);
        }

        // Reset counters for this iteration
        state.reset_inserted_counter(queue);

        // CUDA FLOW (GpuDelaunay.cu:788-831):
        // 1. kerVoteForPoint - vote using CURRENT vert_tet (from prev iteration or init)
        // 2. kerPickWinnerPoint - vertex-parallel winner selection
        // 3. kerNegateInsertedVerts - mark winners
        // 4. thrust_copyIf_Insertable - collect winners
        // 5. thrust::gather - get vertex IDs
        // Note: NO point location before voting! Vertices vote for their current tets.

        // BATCHED: Reset votes → Vote → Pick winner → Build insert list
        // This eliminates 3 submit/poll cycles (was 4 separate submissions, now 1)
        let mut encoder = device.create_command_encoder(&Default::default());

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

        // Debug: Read vert_sphere and tet_vote after voting
        if iteration == 2 {
            let vert_sphere_debug: Vec<i32> = state
                .buffers
                .read_buffer_as(device, queue, &state.buffers.vert_sphere, num_uninserted as usize)
                .await;
            let tet_vote_debug: Vec<i32> = state
                .buffers
                .read_buffer_as(device, queue, &state.buffers.tet_vote, state.max_tets as usize)
                .await;
            println!("[DEBUG] After vote: vert_sphere = {:?}", vert_sphere_debug);
            println!("[DEBUG] After vote: tet_vote (non-NO_VOTE) = {:?}",
                tet_vote_debug.iter().enumerate()
                    .filter(|(_, &v)| v != i32::MIN)
                    .collect::<Vec<_>>());
        }

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

        let mut encoder = device.create_command_encoder(&Default::default());

        // Expand tetrahedron list to make room for new insertions
        // Port of expandTetraList() call from GpuDelaunay.cu:844
        // CRITICAL: Expand by num_inserted (winners this iteration), NOT num_uninserted (all remaining)!
        // CUDA: expandTetraList( &realInsVertVec, ... ) where realInsVertVec.size() == _insNum
        state.expand_tetra_list(&mut encoder, queue, num_inserted, &insert_list);

        // 5c. Split points (update vert_tet for vertices whose tets are splitting)
        state.dispatch_split_points(&mut encoder, queue, num_uninserted);

        // 5d. Split
        println!("[DEBUG] Dispatching split with num_inserted = {}", num_inserted);
        state.dispatch_split(&mut encoder, queue, num_inserted);

        println!("[DEBUG] Submitting batched expand+split_points+split");
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        println!("[DEBUG] Batched operations complete");

        // Debug: Check tet_info and counters after split
        if iteration == 1 {
            let tet_info_debug: Vec<u32> = state
                .buffers
                .read_buffer_as(device, queue, &state.buffers.tet_info, 10)
                .await;
            let counters_debug = state.buffers.read_counters(device, queue).await;
            println!("[DEBUG] After split in iteration 1: tet_info[0..10] = {:?}", tet_info_debug);
            println!("[DEBUG] After split in iteration 1: scratch[2] (threads entered) = {}, scratch[0] (splits run) = {}, scratch[1] (alloc ok) = {}, scratch[3] (alloc fail) = {}",
                counters_debug.scratch[2], counters_debug.scratch[0], counters_debug.scratch[1], counters_debug.scratch[3]);
        }

        // 5e. Split fixup (DISABLED - split.wgsl handles adjacency inline)
        // let mut encoder = device.create_command_encoder(&Default::default());
        // state.dispatch_split_fixup(&mut encoder, queue, num_inserted);
        // queue.submit(Some(encoder.finish()));
        // device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        // Note: We don't update vert_tet here because it will be done at the start of the next iteration

        // 6. Flip checking (optional, iterative) - TWO-PHASE FLIPPING
        if config.enable_flipping {
            // CUDA two-phase flipping:
            // Phase 1: doFlippingLoop(Fast) - f32 predicates for 99%+ of cases
            // Phase 2: markSpecialTets() → doFlippingLoop(Exact) - DD + SoS for degenerate cases

            // Initialize active tet vector with newly split tets (4 per insertion)
            // TODO: Populate act_tet_vec from split operation
            let mut flip_queue_size = num_inserted * 4;

            // Manage vote offset to separate insertion and flip voting
            // CUDA Reference: GpuDelaunay.cu:1121-1128
            let tet_num = state.current_tet_num;
            if state.vote_offset < tet_num {
                // Not enough space - reset tet_vote buffer and offset
                // In CUDA this does: _tetVoteVec.assign(_tetVoteVec.capacity(), INT_MAX)
                // For WGPU, we rely on reset_votes being called each iteration
                state.vote_offset = state.max_tets;
            }
            state.vote_offset -= tet_num;
            let vote_offset = state.vote_offset;

            let use_alternate = false; // TODO: Implement double buffering if needed

            // Track flip batch boundaries for relocateAll
            let mut org_flip_num: Vec<u32> = Vec::new();
            let mut total_flips = 0u32;

            // ========== PHASE 1: FAST FLIPPING (f32 predicates) ==========
            eprintln!("[FLIP] Starting fast flipping phase with {} initial tets", flip_queue_size);

            for flip_iter in 0..config.max_flip_iterations {
                if flip_queue_size == 0 {
                    break;
                }

                // Reset compaction counters
                state.reset_inserted_counter(queue); // Reuse for compaction counters

                // BATCHED: Check → Validate → Compact
                // Eliminates 2 submit/poll cycles per flip iteration
                let mut encoder = device.create_command_encoder(&Default::default());

                // STEP 1: VOTE - Check Delaunay violations and vote for flips
                state.dispatch_check_delaunay_fast(&mut encoder, queue, flip_queue_size, vote_offset);

                // STEP 2: VALIDATE - Mark rejected flips
                state.dispatch_mark_rejected_flips(&mut encoder, queue, flip_queue_size, vote_offset as i32, true);

                // STEP 3: COMPACT - Remove rejected flips
                state.dispatch_compact_if_negative(&mut encoder, queue, flip_queue_size);

                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

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

                // BATCHED: Allocate → Flip → Update
                // Eliminates 2 submit/poll cycles per flip iteration
                let mut encoder = device.create_command_encoder(&Default::default());

                // STEP 4: ALLOCATE - Reserve slots for 2-3 flips
                let inf_idx = state.max_tets; // Infinity vertex at end
                state.dispatch_allocate_flip23_slot(&mut encoder, queue, flip_count, inf_idx, state.max_tets);

                // STEP 5: FLIP - Perform the flips
                let inf_idx = state.num_points;  // Infinity point index
                state.dispatch_flip(&mut encoder, queue, flip_count, inf_idx, use_alternate);

                // STEP 6: UPDATE - Fix adjacency
                state.dispatch_update_opp(&mut encoder, queue, 0, flip_count);

                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

                // Track this batch boundary for relocateAll
                org_flip_num.push(total_flips);
                total_flips += flip_count;

                // Check for flip array overflow
                if total_flips > state.max_flips {
                    eprintln!("[WARNING] Flip array overflow: {} flips > {} capacity",
                             total_flips, state.max_flips);
                    eprintln!("[WARNING] Dataset requires more flips than allocated");
                    eprintln!("[WARNING] Consider increasing allocation or chunking input");
                    break;  // Exit flipping, will fall back to star splaying
                }

                // Read back how many tets were enqueued for the next round
                let new_count = state.buffers.read_flip_count(device, queue).await;
                flip_queue_size = new_count;
            }

            eprintln!("[FLIP] Fast phase complete: {} total flips", total_flips);

            // ========== PHASE 2: EXACT FLIPPING (DD + SoS for degenerate cases) ==========
            // Mark tets that had uncertain predicates (OPP_SPECIAL flag set by fast phase)
            eprintln!("[FLIP] Checking for special tets requiring exact predicates...");
            let mut encoder = device.create_command_encoder(&Default::default());
            state.dispatch_mark_special_tets(&mut encoder, queue, state.max_tets);
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

            // Count how many special tets need exact processing
            let special_tet_count = state.buffers.read_compact_count(device, queue).await;
            eprintln!("[FLIP] Special tet count: {}", special_tet_count);

            if special_tet_count == 0 {
                eprintln!("[FLIP] No special tets - fast phase was sufficient");
            } else {
                eprintln!("[FLIP] Found {} special tets requiring exact predicates", special_tet_count);

                flip_queue_size = special_tet_count;

                for flip_iter in 0..config.max_flip_iterations {
                    if flip_queue_size == 0 {
                        break;
                    }

                    // Reset compaction counters
                    state.reset_inserted_counter(queue);

                    // BATCHED: Check → Validate → Compact
                    // Eliminates 2 submit/poll cycles per flip iteration
                    let mut encoder = device.create_command_encoder(&Default::default());

                    // STEP 1: VOTE - Check Delaunay violations with exact predicates
                    state.dispatch_check_delaunay_exact(&mut encoder, queue, flip_queue_size, vote_offset);

                    // STEP 2: VALIDATE - Mark rejected flips
                    state.dispatch_mark_rejected_flips(&mut encoder, queue, flip_queue_size, vote_offset as i32, true);

                    // STEP 3: COMPACT - Remove rejected flips
                    state.dispatch_compact_if_negative(&mut encoder, queue, flip_queue_size);

                    queue.submit(Some(encoder.finish()));
                    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

                    // Read compacted flip count
                    let flip_count = state.buffers.read_compact_count(device, queue).await;

                    eprintln!(
                        "[FLIP-EXACT] Iteration {}: {} active -> {} valid flips",
                        flip_iter,
                        flip_queue_size,
                        flip_count
                    );

                    if flip_count == 0 {
                        break;
                    }

                    // BATCHED: Allocate → Flip → Update
                    // Eliminates 2 submit/poll cycles per flip iteration
                    let mut encoder = device.create_command_encoder(&Default::default());

                    // STEP 4: ALLOCATE - Reserve slots for 2-3 flips
                    let inf_idx = state.max_tets;
                    state.dispatch_allocate_flip23_slot(&mut encoder, queue, flip_count, inf_idx, state.max_tets);

                    // STEP 5: FLIP - Perform the flips
                    let inf_idx = state.num_points;  // Infinity point index
                    state.dispatch_flip(&mut encoder, queue, flip_count, inf_idx, use_alternate);

                    // STEP 6: UPDATE - Fix adjacency
                    state.dispatch_update_opp(&mut encoder, queue, 0, flip_count);

                    queue.submit(Some(encoder.finish()));
                    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

                    // Track this batch boundary for relocateAll
                    org_flip_num.push(total_flips);
                    total_flips += flip_count;

                    // Check for flip array overflow
                    if total_flips > state.max_flips {
                        eprintln!("[WARNING] Flip array overflow: {} flips > {} capacity",
                                 total_flips, state.max_flips);
                        eprintln!("[WARNING] Dataset requires more flips than allocated");
                        eprintln!("[WARNING] Consider increasing allocation or chunking input");
                        break;  // Exit flipping, will fall back to star splaying
                    }

                    // Read back how many tets were enqueued for the next round
                    let new_count = state.buffers.read_flip_count(device, queue).await;
                    flip_queue_size = new_count;
                }

                eprintln!("[FLIP] Exact phase complete: {} additional flips, {} total", total_flips - org_flip_num.last().copied().unwrap_or(0), total_flips);
            }

            // CRITICAL: Relocate points after flipping (updates vert_tet for points whose tets flipped)
            // Port of GpuDel::relocateAll() from GpuDelaunay.cu:1250-1311
            if total_flips > 0 {
                // Initialize tet_to_flip buffer with -1 (maps tet index → flip chain head)
                let init_data = vec![-1i32; state.max_tets as usize];
                queue.write_buffer(&state.buffers.tet_to_flip, 0, bytemuck::cast_slice(&init_data));

                // BATCHED: Build all flip traces + relocate in single submission
                // Eliminates N submit/poll cycles (where N = number of flip batches)
                let mut encoder = device.create_command_encoder(&Default::default());

                // Build flip traces by iterating batches in REVERSE order
                let mut next_flip_num = total_flips;
                for i in (0..org_flip_num.len()).rev() {
                    let prev_flip_num = org_flip_num[i];
                    let flip_num = next_flip_num - prev_flip_num;

                    state.dispatch_update_flip_trace(&mut encoder, queue, prev_flip_num, flip_num);

                    next_flip_num = prev_flip_num;
                }

                // Now relocate points using the flip trace chains
                state.dispatch_relocate_points_fast(&mut encoder, queue, num_uninserted, total_flips);

                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            }
        }

        // 7. Remove inserted points from uninserted list and compact vert_tet to match.
        // GPU-accelerated compaction (FLAW #2 fix)
        {
            // DEBUG: Verify insert_list before compaction
            let insert_list_before: Vec<[u32; 2]> = state
                .buffers
                .read_buffer_as(device, queue, &state.buffers.insert_list, num_inserted as usize)
                .await;
            println!(
                "[DEBUG_COMPACT] Before compaction: insert_list = {:?}",
                insert_list_before
            );

            // Dispatch GPU compaction (2-pass: count, then scatter)
            // Returns encoder after pass 2 for adding buffer copies
            let mut encoder = state.dispatch_compact_vertex_arrays(device, queue, num_uninserted, num_inserted);

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

        // Read back compacted count and arrays
        let all_counters = state.buffers.read_counters(device, queue).await;
        let new_count = all_counters.free_count as usize; // counter[0]
        println!(
            "[DEBUG_COMPACT] Iteration {}: num_uninserted={}, num_inserted={}, new_count={}",
            iteration, num_uninserted, num_inserted, new_count
        );
        println!(
            "[DEBUG_COMPACT] All counters: free={}, active={}, inserted={}, failed={}",
            all_counters.free_count,
            all_counters.active_count,
            all_counters.inserted_count,
            all_counters.failed_count
        );
        let new_uninserted: Vec<u32> = state
            .buffers
            .read_buffer_as(device, queue, &state.buffers.uninserted, new_count)
            .await;

        // Update CPU state
        state.uninserted = new_uninserted;

        println!(
            "[DEBUG] After iteration {}: {} points remaining (GPU compacted)",
            iteration,
            state.uninserted.len()
        );
    }

    // Final: gather failed vertices
    {
        let mut encoder = device.create_command_encoder(&Default::default());
        state.dispatch_gather(&mut encoder);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
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

    // insert_list is vec2<u32> (tet_idx, vert_idx) — pair[1] is already the vertex ID
    let raw: Vec<[u32; 2]> = state
        .buffers
        .read_buffer_as(device, queue, &state.buffers.insert_list, count)
        .await;

    // Extract vertex IDs directly (no conversion needed)
    raw.iter().map(|pair| pair[1]).collect()
}
