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

        // Debug: Check vert_tet containment at start of iteration
        if iteration >= 2 && iteration <= 6 {
            let num_pos = num_uninserted as usize;
            let vert_tet_data: Vec<u32> = state
                .buffers
                .read_buffer_as(device, queue, &state.buffers.vert_tet, num_pos)
                .await;
            let uninserted_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.uninserted, num_pos).await;
            let tet_data: Vec<[u32; 4]> = state.buffers.read_buffer_as(device, queue, &state.buffers.tets, state.max_tets as usize).await;
            let points_data: Vec<[f32; 4]> = state.buffers.read_buffer_as(device, queue, &state.buffers.points, state.num_points as usize + 5).await;
            let tet_info_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.tet_info, state.max_tets as usize).await;
            let inf_idx = state.num_points + 4;
            let mut ok = 0u32;
            let mut outside = 0u32;
            let mut dead = 0u32;
            let mut invalid = 0u32;
            let mut boundary = 0u32;
            for pos in 0..num_pos {
                let ti = vert_tet_data[pos] as usize;
                if ti == 0xFFFFFFFF { invalid += 1; continue; }
                if ti >= tet_data.len() { invalid += 1; continue; }
                if (tet_info_data[ti] & 1) == 0 { dead += 1; continue; }
                let verts = tet_data[ti];
                if verts[0] >= inf_idx || verts[1] >= inf_idx || verts[2] >= inf_idx || verts[3] >= inf_idx {
                    boundary += 1; continue;
                }
                let vertex_id = uninserted_data[pos];
                let p = [points_data[vertex_id as usize][0], points_data[vertex_id as usize][1], points_data[vertex_id as usize][2]];
                let tv: Vec<[f32; 3]> = (0..4).map(|i| {
                    let vi = verts[i] as usize;
                    [points_data[vi][0], points_data[vi][1], points_data[vi][2]]
                }).collect();
                let faces = [[1,3,2],[0,2,3],[0,3,1],[0,1,2]];
                let tet_orient = {
                    let ad = [tv[0][0]-tv[3][0], tv[0][1]-tv[3][1], tv[0][2]-tv[3][2]];
                    let bd = [tv[1][0]-tv[3][0], tv[1][1]-tv[3][1], tv[1][2]-tv[3][2]];
                    let cd = [tv[2][0]-tv[3][0], tv[2][1]-tv[3][1], tv[2][2]-tv[3][2]];
                    ad[0]*(bd[1]*cd[2]-bd[2]*cd[1]) + bd[0]*(cd[1]*ad[2]-cd[2]*ad[1]) + cd[0]*(ad[1]*bd[2]-ad[2]*bd[1])
                };
                let mut is_inside = true;
                for face in &faces {
                    let a = tv[face[0]]; let b = tv[face[1]]; let c = tv[face[2]];
                    let ad = [a[0]-p[0], a[1]-p[1], a[2]-p[2]];
                    let bd = [b[0]-p[0], b[1]-p[1], b[2]-p[2]];
                    let cd = [c[0]-p[0], c[1]-p[1], c[2]-p[2]];
                    let det = ad[0]*(bd[1]*cd[2]-bd[2]*cd[1]) + bd[0]*(cd[1]*ad[2]-cd[2]*ad[1]) + cd[0]*(ad[1]*bd[2]-ad[2]*bd[1]);
                    if tet_orient < 0.0 && det > 0.0 { is_inside = false; }
                    if tet_orient > 0.0 && det < 0.0 { is_inside = false; }
                }
                if is_inside { ok += 1; } else { outside += 1; }
            }
            eprintln!("[VERT_TET_CHECK] Iter {}: {} inside, {} OUTSIDE, {} dead, {} invalid, {} boundary (of {} uninserted)",
                iteration, ok, outside, dead, invalid, boundary, num_pos);
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
            println!("[DEBUG] No winners found in iteration {} with {} uninserted — breaking", iteration, num_uninserted);
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

        // DEBUG: Check if inserted vertices are actually inside their assigned tets
        {
            let uninserted_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.uninserted, num_uninserted as usize).await;
            let tet_data: Vec<[u32; 4]> = state.buffers.read_buffer_as(device, queue, &state.buffers.tets, state.max_tets as usize).await;
            let points_data: Vec<[f32; 4]> = state.buffers.read_buffer_as(device, queue, &state.buffers.points, state.num_points as usize + 5).await;
            let tet_info_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.tet_info, state.max_tets as usize).await;
            let inf_idx = state.num_points + 4;
            let mut inside_count = 0u32;
            let mut outside_count = 0u32;
            let mut boundary_count = 0u32;
            let mut dead_count = 0u32;
            for entry in &insert_list {
                let tet_idx = entry[0] as usize;
                let position = entry[1] as usize;
                if tet_idx >= tet_data.len() { continue; }
                if (tet_info_data[tet_idx] & 1) == 0 {
                    dead_count += 1;
                    continue;
                }
                let verts = tet_data[tet_idx];
                if verts[0] >= inf_idx || verts[1] >= inf_idx || verts[2] >= inf_idx || verts[3] >= inf_idx {
                    boundary_count += 1;
                    continue;
                }
                let vertex_id = if position < uninserted_data.len() { uninserted_data[position] } else { continue; };
                let p = [points_data[vertex_id as usize][0], points_data[vertex_id as usize][1], points_data[vertex_id as usize][2]];
                let tv: Vec<[f32; 3]> = (0..4).map(|i| {
                    let vi = verts[i] as usize;
                    [points_data[vi][0], points_data[vi][1], points_data[vi][2]]
                }).collect();
                // Check orient3d for all 4 faces with the vertex substituted
                // TetViAsSeenFrom: face i is opposite vertex i
                // For vertex inside tet with orient3d < 0, orient3d(face_verts, vertex) should be < 0
                let faces = [[1,3,2],[0,2,3],[0,3,1],[0,1,2]]; // TetViAsSeenFrom
                let tet_orient = {
                    let ad = [tv[0][0]-tv[3][0], tv[0][1]-tv[3][1], tv[0][2]-tv[3][2]];
                    let bd = [tv[1][0]-tv[3][0], tv[1][1]-tv[3][1], tv[1][2]-tv[3][2]];
                    let cd = [tv[2][0]-tv[3][0], tv[2][1]-tv[3][1], tv[2][2]-tv[3][2]];
                    ad[0]*(bd[1]*cd[2]-bd[2]*cd[1]) + bd[0]*(cd[1]*ad[2]-cd[2]*ad[1]) + cd[0]*(ad[1]*bd[2]-ad[2]*bd[1])
                };
                let mut is_inside = true;
                for face in &faces {
                    let a = tv[face[0]];
                    let b = tv[face[1]];
                    let c = tv[face[2]];
                    let ad = [a[0]-p[0], a[1]-p[1], a[2]-p[2]];
                    let bd = [b[0]-p[0], b[1]-p[1], b[2]-p[2]];
                    let cd = [c[0]-p[0], c[1]-p[1], c[2]-p[2]];
                    let det = ad[0]*(bd[1]*cd[2]-bd[2]*cd[1]) + bd[0]*(cd[1]*ad[2]-cd[2]*ad[1]) + cd[0]*(ad[1]*bd[2]-ad[2]*bd[1]);
                    // For tet with orient3d < 0: vertex inside means all face orient3d < 0
                    // For tet with orient3d > 0: vertex inside means all face orient3d > 0
                    if tet_orient < 0.0 && det > 0.0 { is_inside = false; }
                    if tet_orient > 0.0 && det < 0.0 { is_inside = false; }
                }
                if is_inside {
                    inside_count += 1;
                } else {
                    outside_count += 1;
                    if outside_count <= 3 {
                        // Print detailed info for first 3 OUTSIDE points
                        let mut face_dets = [0.0f32; 4];
                        for (fi, face) in faces.iter().enumerate() {
                            let a = tv[face[0]];
                            let b = tv[face[1]];
                            let c = tv[face[2]];
                            let ad = [a[0]-p[0], a[1]-p[1], a[2]-p[2]];
                            let bd = [b[0]-p[0], b[1]-p[1], b[2]-p[2]];
                            let cd = [c[0]-p[0], c[1]-p[1], c[2]-p[2]];
                            face_dets[fi] = ad[0]*(bd[1]*cd[2]-bd[2]*cd[1]) + bd[0]*(cd[1]*ad[2]-cd[2]*ad[1]) + cd[0]*(ad[1]*bd[2]-ad[2]*bd[1]);
                        }
                        eprintln!("[OUTSIDE] pos={}, vert={}, tet={}, verts={:?}, orient={:.6e}, face_dets=[{:.6e}, {:.6e}, {:.6e}, {:.6e}]",
                            position, vertex_id, tet_idx, verts, tet_orient, face_dets[0], face_dets[1], face_dets[2], face_dets[3]);
                    }
                }
            }
            if outside_count > 0 || dead_count > 0 {
                eprintln!("[CONTAINMENT] Iter {}: {} inside, {} OUTSIDE, {} boundary, {} dead (of {} inserts)",
                    iteration, inside_count, outside_count, boundary_count, dead_count, num_inserted);
            }
        }

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

        // 5b. Write base_tet to tet_to_vert for concurrent split detection and split_points.
        // Each insertion idx gets tets at [next_free_tet + idx*4 .. +3].
        // By writing the base_tet to tet_to_vert[old_tet] BEFORE split, all threads
        // can detect concurrent splits by reading tet_to_vert[nei_tet] which gives
        // the neighbor's base_tet. This avoids CUDA's free_arr-based lookup which
        // doesn't work with our deterministic allocation scheme.
        for (idx, entry) in insert_list.iter().enumerate() {
            let tet_idx = entry[0];
            let base_tet = state.next_free_tet + (idx as u32) * 4;
            queue.write_buffer(
                &state.buffers.tet_to_vert,
                (tet_idx as u64) * 4,
                bytemuck::cast_slice(&[base_tet]),
            );
        }

        // 5c. Split (allocate new tets, write vertex data, mark old dead)
        // tet_to_vert[old_tet] already contains base_tet (written above).
        // split reads it for concurrent detection; split_points reads it
        // after split to find new tet locations.
        state.cpu_profiler.begin("4_split");
        println!("[DEBUG] Dispatching split with num_inserted = {}", num_inserted);
        let mut encoder = device.create_command_encoder(&Default::default());
        state.dispatch_split(&mut encoder, queue, num_inserted);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        // Advance global tet counter after split (each insertion allocates 4 tets)
        state.next_free_tet += num_inserted * 4;
        println!("[DEBUG] Batched operations complete (next_free_tet={})", state.next_free_tet);
        state.cpu_profiler.end("4_split");

        // 5d. Split points (update vert_tet for vertices whose tets were just split)
        // Reads base_tet from tet_to_vert[tet_idx] (written by split above) and
        // reconstructs original 5-vertex config from new tet data for the decision tree.
        state.cpu_profiler.begin("3_split_points");
        let mut encoder = device.create_command_encoder(&Default::default());
        state.dispatch_split_points(&mut encoder, queue, num_uninserted);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        state.cpu_profiler.end("3_split_points");

        // DEBUG: Check vert_tet after split_points
        if iteration <= 3 {
            let num_pos = num_uninserted as usize;
            let vt: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.vert_tet, num_pos).await;
            let tet_info_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.tet_info, state.max_tets as usize).await;
            let mut hist = std::collections::HashMap::new();
            let mut dead_count = 0u32;
            let mut invalid_count = 0u32;
            for pos in 0..num_pos {
                let ti = vt[pos];
                if ti == 0xFFFFFFFF { invalid_count += 1; continue; }
                if (ti as usize) < tet_info_data.len() && (tet_info_data[ti as usize] & 1) == 0 {
                    dead_count += 1;
                }
                *hist.entry(ti).or_insert(0u32) += 1;
            }
            let mut top: Vec<_> = hist.into_iter().collect();
            top.sort_by(|a, b| b.1.cmp(&a.1));
            top.truncate(10);
            eprintln!("[SPLIT_PTS_DBG] Iter {}: {} positions, {} dead, {} invalid, top tets: {:?}",
                iteration, num_pos, dead_count, invalid_count, top);

            // CPU simulation of decision tree for iteration 1
            if iteration == 1 {
                let tet_data: Vec<[u32; 4]> = state.buffers.read_buffer_as(device, queue, &state.buffers.tets, state.max_tets as usize).await;
                let points_data: Vec<[f32; 4]> = state.buffers.read_buffer_as(device, queue, &state.buffers.points, state.num_points as usize + 5).await;
                let uninserted_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.uninserted, num_pos).await;
                let tet_to_vert_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.tet_to_vert, state.max_tets as usize).await;

                let split_faces: [[usize; 3]; 11] = [
                    [0,1,4],[0,3,4],[0,2,4],[2,3,4],[1,3,4],[1,2,4],[2,3,4],
                    [1,3,2],[0,2,3],[0,3,1],[0,1,2]
                ];
                let split_next: [[usize; 2]; 11] = [
                    [1,2],[3,4],[5,6],[7,8],[9,7],[7,10],[7,8],
                    [1,0],[2,0],[3,0],[4,0]
                ];

                fn orient3d_f64(a: [f64;3], b: [f64;3], c: [f64;3], d: [f64;3]) -> f64 {
                    let ad = [a[0]-d[0], a[1]-d[1], a[2]-d[2]];
                    let bd = [b[0]-d[0], b[1]-d[1], b[2]-d[2]];
                    let cd = [c[0]-d[0], c[1]-d[1], c[2]-d[2]];
                    ad[0]*(bd[1]*cd[2]-bd[2]*cd[1]) + bd[0]*(cd[1]*ad[2]-cd[2]*ad[1]) + cd[0]*(ad[1]*bd[2]-ad[2]*bd[1])
                }

                // For vertex at pos 0
                let pos = 0usize;
                let old_tet = 0usize; // vert_tet should be 0 initially
                let base_tet = tet_to_vert_data[old_tet];
                eprintln!("  [CPU-TREE] pos={}, vert={}, old_tet={}, base_tet={}", pos, uninserted_data[pos], old_tet, base_tet);

                if base_tet != 0xFFFFFFFF {
                    let bt = base_tet as usize;
                    let t0 = tet_data[bt]; // (v1, v3, v2, split)
                    let t1 = tet_data[bt+1]; // (v0, v2, v3, split)
                    let split_vertex = t0[3];
                    // Reconstruct: {v0, v1, v2, v3, split}
                    let vert5 = [t1[0], t0[0], t0[2], t0[1], split_vertex];
                    let pts5: Vec<[f64;3]> = vert5.iter().map(|&vi| {
                        let p = points_data[vi as usize];
                        [p[0] as f64, p[1] as f64, p[2] as f64]
                    }).collect();
                    let vid = uninserted_data[pos] as usize;
                    let pt = [points_data[vid][0] as f64, points_data[vid][1] as f64, points_data[vid][2] as f64];

                    eprintln!("  [CPU-TREE] vert5={:?}", vert5);
                    let mut face = 0usize;
                    for iter in 0..3 {
                        let fv = split_faces[face];
                        let o = orient3d_f64(pts5[fv[0]], pts5[fv[1]], pts5[fv[2]], pt);
                        let branch = if o < 0.0 { 0 } else { 1 }; // orient < 0 → index 0 (OrientPos)
                        let next = split_next[face][branch];
                        eprintln!("  [CPU-TREE] iter={}, face={}, fv={:?}, orient={:.6e}, branch={}, next={}",
                            iter, face, fv, o, branch, next);
                        face = next;
                    }
                    let cpu_offset = if face >= 7 && face <= 10 { face - 7 } else { 999 };
                    let cpu_tet = if cpu_offset < 4 { base_tet + cpu_offset as u32 } else { 0xFFFFFFFF };
                    let gpu_tet = vt[pos];
                    eprintln!("  [CPU-TREE] final face={}, offset={}, cpu_tet={}, gpu_tet={}, match={}",
                        face, cpu_offset, cpu_tet, gpu_tet, cpu_tet == gpu_tet);

                    // Check containment for all 4 sub-tets
                    for sub in 0..4u32 {
                        let ti = (base_tet + sub) as usize;
                        let tv = tet_data[ti];
                        let verts: Vec<[f64;3]> = (0..4).map(|i| {
                            let vi = tv[i] as usize;
                            [points_data[vi][0] as f64, points_data[vi][1] as f64, points_data[vi][2] as f64]
                        }).collect();
                        let tet_o = orient3d_f64(verts[0], verts[1], verts[2], verts[3]);
                        let faces_idx = [[1,3,2],[0,2,3],[0,3,1],[0,1,2]];
                        let mut inside = true;
                        for fi in &faces_idx {
                            let fo = orient3d_f64(verts[fi[0]], verts[fi[1]], verts[fi[2]], pt);
                            if (tet_o < 0.0 && fo > 0.0) || (tet_o > 0.0 && fo < 0.0) {
                                inside = false;
                            }
                        }
                        eprintln!("  [CPU-TREE] sub-tet {} (tet {}): orient={:.4e}, inside={}", sub, ti, tet_o, inside);
                    }
                }
            }
        }

        // 5e. Split fixup (DISABLED - split.wgsl handles adjacency inline)
        // let mut encoder = device.create_command_encoder(&Default::default());
        // state.dispatch_split_fixup(&mut encoder, queue, num_inserted);
        // queue.submit(Some(encoder.finish()));
        // device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        // Note: We don't update vert_tet here because it will be done at the start of the next iteration

        // DEBUG: Verify adjacency consistency after split (read full buffer)
        {
            let tet_num = state.max_tets as usize;
            let tet_info_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.tet_info, tet_num).await;
            let tet_opp_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.tet_opp, tet_num * 4).await;
            let tet_data: Vec<[u32; 4]> = state.buffers.read_buffer_as(device, queue, &state.buffers.tets, tet_num).await;
            let mut dead_refs = 0u32;
            let mut oob_refs = 0u32;
            let mut back_opp_mismatch = 0u32;
            let mut internal_count = 0u32;
            let mut geom_wrong = 0u32;
            let mut printed_geom = 0u32;
            let face_vi_map: [[usize; 3]; 4] = [[1,2,3],[0,2,3],[0,1,3],[0,1,2]];
            let alive_count = tet_info_data.iter().filter(|&&x| (x & 1) != 0).count();
            for t in 0..tet_num {
                if (tet_info_data[t] & 1) == 0 { continue; }
                for f in 0..4usize {
                    let opp = tet_opp_data[t * 4 + f];
                    if opp == 0xFFFFFFFF { continue; }
                    if (opp & 4) != 0 { internal_count += 1; }
                    let nei_tet = (opp >> 5) as usize;
                    let nei_face = (opp & 3) as usize;
                    if nei_tet >= tet_num {
                        oob_refs += 1;
                    } else if (tet_info_data[nei_tet] & 1) == 0 {
                        dead_refs += 1;
                    } else {
                        let back_opp = tet_opp_data[nei_tet * 4 + nei_face];
                        if back_opp != 0xFFFFFFFF {
                            let back_tet = (back_opp >> 5) as usize;
                            let back_face = (back_opp & 3) as usize;
                            if back_tet != t || back_face != f {
                                back_opp_mismatch += 1;
                            }
                        }
                        // Geometric face check: faces should share exactly 3 vertices
                        if t < tet_data.len() && nei_tet < tet_data.len() {
                            let tv = tet_data[t];
                            let nv = tet_data[nei_tet];
                            let fa: [u32; 3] = [tv[face_vi_map[f][0]], tv[face_vi_map[f][1]], tv[face_vi_map[f][2]]];
                            let fb: [u32; 3] = [nv[face_vi_map[nei_face][0]], nv[face_vi_map[nei_face][1]], nv[face_vi_map[nei_face][2]]];
                            let shared = fa.iter().filter(|v| fb.contains(v)).count();
                            if shared != 3 {
                                geom_wrong += 1;
                                if printed_geom < 5 {
                                    printed_geom += 1;
                                    let is_internal = (opp & 4) != 0;
                                    eprintln!("  GEOM-WRONG: tet[{}].f{} ({:?}) -> tet[{}].f{} ({:?}), shared={}, internal={}",
                                        t, f, fa, nei_tet, nei_face, fb, shared, is_internal);
                                }
                            }
                        }
                    }
                }
            }
            eprintln!("[ADJACENCY-CHECK] Iter {} post-split: alive={}, dead_refs={}, oob_refs={}, back_opp_mismatch={}, geom_wrong={}, internal={}",
                iteration, alive_count, dead_refs, oob_refs, back_opp_mismatch, geom_wrong, internal_count);
        }

        // DEBUG: Orient check after split, before flips
        {
            let tet_num = state.current_tet_num as usize;
            let points_data: Vec<[f32; 4]> = state.buffers.read_buffer_as(device, queue, &state.buffers.points, state.num_points as usize + 5).await;
            let tet_data: Vec<[u32; 4]> = state.buffers.read_buffer_as(device, queue, &state.buffers.tets, tet_num).await;
            let tet_info_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.tet_info, tet_num).await;
            let inf_idx = state.num_points + 4;
            let mut pos_count = 0u32;
            let mut neg_count = 0u32;
            for t in 0..tet_num {
                if (tet_info_data[t] & 1) == 0 { continue; }
                let v = tet_data[t];
                if v[0] >= inf_idx || v[1] >= inf_idx || v[2] >= inf_idx || v[3] >= inf_idx { continue; }
                let a = [points_data[v[0] as usize][0], points_data[v[0] as usize][1], points_data[v[0] as usize][2]];
                let b = [points_data[v[1] as usize][0], points_data[v[1] as usize][1], points_data[v[1] as usize][2]];
                let c = [points_data[v[2] as usize][0], points_data[v[2] as usize][1], points_data[v[2] as usize][2]];
                let d = [points_data[v[3] as usize][0], points_data[v[3] as usize][1], points_data[v[3] as usize][2]];
                let ad = [a[0]-d[0], a[1]-d[1], a[2]-d[2]];
                let bd = [b[0]-d[0], b[1]-d[1], b[2]-d[2]];
                let cd = [c[0]-d[0], c[1]-d[1], c[2]-d[2]];
                let det = ad[0]*(bd[1]*cd[2]-bd[2]*cd[1]) + bd[0]*(cd[1]*ad[2]-cd[2]*ad[1]) + cd[0]*(ad[1]*bd[2]-ad[2]*bd[1]);
                if det > 0.0 { pos_count += 1; } else if det < 0.0 { neg_count += 1; }
            }
            eprintln!("[ORIENT-POST-SPLIT] Iter {}: {} positive, {} negative orient3d (BEFORE flips)", iteration, pos_count, neg_count);
        }

        // 6. Flip checking (optional, iterative) - CUDA-STYLE PIPELINE
        // CUDA pipeline: check_delaunay → mark_rejected_flips → flip → update_opp
        // Reference: GpuDelaunay.cu:700-760
        if config.enable_flipping {
            let mut tet_num_for_collect = state.current_tet_num;
            let mut total_flips = 0u32;
            // Track flip batch boundaries for relocateAll (CUDA: _orgFlipNum)
            let mut org_flip_nums: Vec<(u32, u32)> = Vec::new();

            // CRITICAL: Reset tet_msg_arr to (-1, -1) before each flipping phase.
            // CUDA: _tetMsgVec.assign( _tetVec.size(), make_int2( -1, -1 ) )
            // (GpuDelaunay.cu:1090 and 1306)
            // Without this, stale msg.y values from previous iterations cause update_opp
            // to read garbage flip_arr entries (msg.y >= org_flip_num = 0 for new iteration).
            {
                let fill_size = (state.max_tets as usize) * 8; // vec2<i32> = 8 bytes
                let fill_data = vec![0xFFu8; fill_size];
                queue.write_buffer(&state.buffers.tet_msg_arr, 0, &fill_data);
            }

            // DEBUG: Check free list health before flip phase
            if iteration <= 5 {
                let tet_num = state.max_tets as usize;
                let tet_info_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.tet_info, tet_num).await;
                let free_arr_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.free_arr, tet_num).await;
                let num_verts = (state.num_points + 5) as usize;
                let vert_free_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.vert_free_arr, num_verts).await;
                let mut alive_in_free = 0u32;
                for v in 0..num_verts {
                    let free_count = vert_free_data[v];
                    if free_count == 0 || free_count > 8 { continue; }
                    for s in 0..free_count as usize {
                        let idx = v * 8 + s;
                        if idx >= free_arr_data.len() { continue; }
                        let tet_idx = free_arr_data[idx] as usize;
                        if tet_idx < tet_num && (tet_info_data[tet_idx] & 1) != 0 {
                            alive_in_free += 1;
                            if alive_in_free <= 5 {
                                eprintln!("  FREE-LIST-BUG iter{}: vertex {} slot {} -> tet[{}] is ALIVE (info=0x{:x})",
                                    iteration, v, s, tet_idx, tet_info_data[tet_idx]);
                            }
                        }
                    }
                }
                if alive_in_free > 0 {
                    eprintln!("[FREE-LIST-CHECK] iter{}: {} alive tets found in free list!", iteration, alive_in_free);
                } else {
                    eprintln!("[FREE-LIST-CHECK] iter{}: OK (no alive tets in free list)", iteration);
                }
            }

            // Write next_free_tet to vert_free_arr[0] for flip's allocate_flip23_slot
            queue.write_buffer(&state.buffers.vert_free_arr, 0, bytemuck::cast_slice(&[state.next_free_tet]));

            // ========== FAST FLIP PHASE ==========
            // CUDA pattern: doFlippingLoop calls doFlipping repeatedly.
            // Each doFlipping re-collects active tets, checks, votes, flips, updates opp.
            // Reference: GpuDelaunay.cu:915-1142 (doFlipping), 700-710 (doFlippingLoop)
            state.cpu_profiler.begin("5_flip_fast");

            for flip_iter in 0..config.max_flip_iterations {
                // Re-collect active (alive & changed) tets each iteration — CUDA pattern
                let act_tet_num = state
                    .dispatch_collect_active_tets(device, queue, tet_num_for_collect)
                    .await;
                if act_tet_num == 0 { break; }

                queue.write_buffer(&state.buffers.counters, 0, bytemuck::cast_slice(&[0u32]));
                {
                    let mut encoder = device.create_command_encoder(&Default::default());
                    state.dispatch_reset_flip_votes(&mut encoder, queue);
                    state.dispatch_check_delaunay_fast(&mut encoder, queue, act_tet_num, 0);
                    state.dispatch_mark_rejected_flips(&mut encoder, queue, act_tet_num, 0, true);
                    queue.submit(Some(encoder.finish()));
                    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
                }

                let validated_flips = state.buffers.read_collection_count(device, queue).await;
                if validated_flips == 0 {
                    eprintln!("[FLIP] Fast iter {}: active={}, no validated flips, breaking", flip_iter + 1, act_tet_num);
                    break;
                }

                state.reset_flip_count(queue);
                queue.write_buffer(&state.buffers.counters, 16, bytemuck::cast_slice(&[0u32, 0u32, 0u32, 0u32]));
                {
                    let mut encoder = device.create_command_encoder(&Default::default());
                    encoder.copy_buffer_to_buffer(
                        &state.buffers.flip_to_tet, 0,
                        &state.buffers.flip_queue, 0,
                        (validated_flips as u64) * 4,
                    );
                    let inf_idx = state.num_points + 4;
                    state.dispatch_flip(&mut encoder, queue, validated_flips, inf_idx, false, total_flips);
                    queue.submit(Some(encoder.finish()));
                    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
                }

                let (_next_count, metadata_flips) = state.buffers.read_flip_count(device, queue).await;

                let debug_counters = state.buffers.read_counters(device, queue).await;
                eprintln!(
                    "[FLIP-DBG] invalid_opp={}, dead_b={}, lock_or_dead_a={}, reval_fail={}",
                    debug_counters.scratch[0], debug_counters.scratch[1],
                    debug_counters.scratch[2], debug_counters.scratch[3]
                );

                if metadata_flips > 0 {
                    // DEBUG: Check flip_arr data BEFORE update_opp (isolate flip vs update_opp bugs)
                    if iteration <= 5 {
                        let tet_num = state.max_tets as usize;
                        let tet_data: Vec<[u32; 4]> = state.buffers.read_buffer_as(device, queue, &state.buffers.tets, tet_num).await;
                        let flip_arr_data: Vec<[i32; 4]> = state.buffers.read_buffer_as(device, queue, &state.buffers.flip_arr, (total_flips as usize + metadata_flips as usize) * 2).await;

                        // Dump ALL flip entries and find slot conflicts
                        let mut slot_map: std::collections::HashMap<usize, Vec<(usize, &str)>> = std::collections::HashMap::new();
                        for fi in 0..metadata_flips as usize {
                            let glob_fi = total_flips as usize + fi;
                            let item0 = flip_arr_data[glob_fi * 2];
                            let item1 = flip_arr_data[glob_fi * 2 + 1];
                            let t0 = item1[1] as u32 as usize;
                            let t1 = item1[2] as u32 as usize;
                            let t2_raw = item1[3];
                            let is_flip23 = t2_raw >= 0;
                            let c0 = [item0[0] as u32, item0[1] as u32, item0[2] as u32, item0[3] as u32];

                            eprintln!("  FLIP-ENTRY iter{} flip{} fi={}: c0={:?} v4={} t=[{},{},{}] is23={} actual_t0={:?}",
                                iteration, flip_iter+1, fi, c0, item1[0], t0, t1,
                                if is_flip23 { t2_raw as u32 as usize } else { (-(t2_raw + 2)) as u32 as usize },
                                is_flip23,
                                if t0 < tet_data.len() { Some(tet_data[t0]) } else { None });

                            slot_map.entry(t0).or_default().push((fi, "t0"));
                            slot_map.entry(t1).or_default().push((fi, "t1"));
                            if is_flip23 {
                                slot_map.entry(t2_raw as u32 as usize).or_default().push((fi, "t2"));
                            }
                        }

                        // Report slot conflicts
                        for (slot, users) in &slot_map {
                            if users.len() > 1 {
                                let desc: Vec<String> = users.iter().map(|(fi, role)| format!("fi={} as {}", fi, role)).collect();
                                eprintln!("  SLOT-CONFLICT iter{} flip{}: tet[{}] used by: {}",
                                    iteration, flip_iter+1, slot, desc.join(", "));
                            }
                        }
                    }

                    // Record flip batch boundary BEFORE update_opp
                    org_flip_nums.push((total_flips, metadata_flips));

                    let mut encoder = device.create_command_encoder(&Default::default());
                    state.dispatch_update_opp(&mut encoder, queue, total_flips, metadata_flips);
                    // No donation needed: global atomic counter allocates fresh slots only,
                    // no recycling. Pre-allocated (num_points+5)*8 tets is sufficient.
                    queue.submit(Some(encoder.finish()));
                    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
                }

                eprintln!(
                    "[FLIP] Fast iter {}: active={}, validated={}, executed={}",
                    flip_iter + 1, act_tet_num, validated_flips, metadata_flips
                );

                // DEBUG: Per-flip-batch adjacency check (only first few insertion iters)
                if iteration <= 5 && metadata_flips > 0 {
                    let tet_num = state.max_tets as usize;
                    let tet_info_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.tet_info, tet_num).await;
                    let tet_opp_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.tet_opp, tet_num * 4).await;
                    let tet_data: Vec<[u32; 4]> = state.buffers.read_buffer_as(device, queue, &state.buffers.tets, tet_num).await;
                    let mut geom_wrong_fb = 0u32;
                    let mut dead_refs_fb = 0u32;
                    let mut back_mismatch_fb = 0u32;
                    for t in 0..tet_num {
                        if (tet_info_data[t] & 1) == 0 { continue; }
                        for f in 0..4usize {
                            let opp = tet_opp_data[t * 4 + f];
                            if opp == 0xFFFFFFFF { continue; }
                            let nei_tet = (opp >> 5) as usize;
                            let nei_face = (opp & 3) as usize;
                            if nei_tet >= tet_num { continue; }
                            if (tet_info_data[nei_tet] & 1) == 0 { dead_refs_fb += 1; continue; }
                            // Geometric check
                            let face_vi: [[usize; 3]; 4] = [[1,2,3],[0,2,3],[0,1,3],[0,1,2]];
                            let tv = tet_data[t]; let nv = tet_data[nei_tet];
                            let fa = [tv[face_vi[f][0]], tv[face_vi[f][1]], tv[face_vi[f][2]]];
                            let fb = [nv[face_vi[nei_face][0]], nv[face_vi[nei_face][1]], nv[face_vi[nei_face][2]]];
                            let shared = fa.iter().filter(|v| fb.contains(v)).count();
                            if shared != 3 {
                                geom_wrong_fb += 1;
                                if geom_wrong_fb <= 3 {
                                    let is_int = (opp & 4) != 0;
                                    eprintln!("  GEOM-WRONG iter{} flip{}: tet[{}].f{}({:?})->tet[{}].f{}({:?}) shared={} int={}",
                                        iteration, flip_iter+1, t, f, fa, nei_tet, nei_face, fb, shared, is_int);
                                }
                            }
                            // Back-opp check
                            let back = tet_opp_data[nei_tet * 4 + nei_face];
                            if back != 0xFFFFFFFF {
                                let bt = (back >> 5) as usize;
                                let bf = (back & 3) as usize;
                                if bt != t || bf != f { back_mismatch_fb += 1; }
                            }
                        }
                    }
                    if geom_wrong_fb > 0 || dead_refs_fb > 0 || back_mismatch_fb > 0 {
                        eprintln!("[FLIP-BATCH-CHECK] iter{} flip{}: org_flip={}, batch={}, geom_wrong={}, dead_refs={}, back_mismatch={}",
                            iteration, flip_iter+1, total_flips, metadata_flips, geom_wrong_fb, dead_refs_fb, back_mismatch_fb);
                    }
                }

                total_flips += metadata_flips;

                // Update collection range: 2-3 flips allocate new tets
                // Read back next_free_tet to ensure we scan them next iteration
                if metadata_flips > 0 {
                    let vfa: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.vert_free_arr, 1).await;
                    state.next_free_tet = vfa[0];
                    tet_num_for_collect = state.next_free_tet.max(tet_num_for_collect);
                }

                if metadata_flips == 0 { break; }

                if total_flips > state.max_flips {
                    eprintln!("[WARNING] Flip array overflow: {} > {} capacity",
                             total_flips, state.max_flips);
                    break;
                }
            }

            eprintln!("[FLIP] Fast phase complete: {} total flipped tets", total_flips);
            state.cpu_profiler.end("5_flip_fast");

            // Read back next_free_tet BEFORE exact phase so we scan all tets
            // including those allocated by 2-3 flips during the fast phase.
            {
                let vfa: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.vert_free_arr, 1).await;
                state.next_free_tet = vfa[0];
                eprintln!("[FLIP] next_free_tet after fast phase: {}", state.next_free_tet);
            }

            // ========== MARK SPECIAL TETS ==========
            {
                let tet_num = state.next_free_tet.max(state.current_tet_num);
                let mut encoder = device.create_command_encoder(&Default::default());
                state.dispatch_mark_special_tets(&mut encoder, queue, tet_num);
                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
                eprintln!("[FLIP] Cleared OPP_SPECIAL flags before exact phase");
            }

            // ========== EXACT FLIP PHASE ==========
            // Same doFlippingLoop pattern: re-collect ALL alive tets each iteration
            // CUDA: doFlippingLoop(SphereExactOrientSoS)
            {
                state.cpu_profiler.begin("6_reflip_all");
                // Use next_free_tet as upper bound: 2-3 flips allocate beyond current_tet_num
                let mut tet_num = state.next_free_tet.max(state.current_tet_num);

                for flip_iter in 0..config.max_flip_iterations {
                    let act_tet_num = state
                        .dispatch_collect_all_alive_tets(device, queue, tet_num)
                        .await;
                    if act_tet_num == 0 { break; }

                    // Reset ALL 8 counters for check_delaunay_exact debug counters
                    // (split_points.wgsl and flip.wgsl also write to counters[4..7])
                    queue.write_buffer(&state.buffers.counters, 0, bytemuck::cast_slice(&[0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32]));
                    {
                        let mut encoder = device.create_command_encoder(&Default::default());
                        state.dispatch_reset_flip_votes(&mut encoder, queue);
                        state.dispatch_check_delaunay_exact(&mut encoder, queue, act_tet_num, 0);
                        state.dispatch_mark_rejected_flips(&mut encoder, queue, act_tet_num, 0, true);
                        queue.submit(Some(encoder.finish()));
                        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
                    }

                    let validated_flips = state.buffers.read_collection_count(device, queue).await;

                    // Read debug counters from exact check: [3]=sphere_fail, [4]=32_proposals, [5]=23_proposals
                    let dbg_counters = state.buffers.read_counters(device, queue).await;
                    if validated_flips == 0 {
                        eprintln!("[FLIP-ALL] Iter {}: active={}, no validated flips (sphere_fail={}, 32_prop={}, 23_prop={}, 23_orient_fail={}), breaking",
                            flip_iter + 1, act_tet_num,
                            dbg_counters.failed_count, dbg_counters.scratch[0], dbg_counters.scratch[1], dbg_counters.scratch[2]);

                        // CPU-SIDE VIOLATION ANALYSIS: dump first few violations for debugging
                        if dbg_counters.failed_count > 0 {
                            let tn = tet_num as usize;
                            let tet_data: Vec<[u32; 4]> = state.buffers.read_buffer_as(device, queue, &state.buffers.tets, tn).await;
                            let tet_opp_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.tet_opp, tn * 4).await;
                            let tet_info_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.tet_info, tn).await;
                            let points_data: Vec<[f32; 4]> = state.buffers.read_buffer_as(device, queue, &state.buffers.points, state.num_points as usize + 5).await;
                            let inf_idx = state.num_points + 4;
                            let mut printed = 0u32;
                            for t in 0..tn {
                                if printed >= 3 { break; }
                                if (tet_info_data[t] & 1) == 0 { continue; }
                                let tv = tet_data[t];
                                for f in 0..4u32 {
                                    let opp = tet_opp_data[t * 4 + f as usize];
                                    if opp == 0xFFFFFFFF { continue; }
                                    if (opp & 4) != 0 { continue; } // internal
                                    let nei = (opp >> 5) as usize;
                                    let nei_f = (opp & 3) as usize;
                                    if nei >= tn { continue; }
                                    if (tet_info_data[nei] & 1) == 0 { continue; }
                                    let nv = tet_data[nei];
                                    // Get opposite vertex
                                    let top_vert = match nei_f { 0 => nv[0], 1 => nv[1], 2 => nv[2], _ => nv[3] };
                                    if top_vert >= inf_idx { continue; }
                                    if tv[0] >= inf_idx || tv[1] >= inf_idx || tv[2] >= inf_idx || tv[3] >= inf_idx { continue; }
                                    // Simple insphere check
                                    let a = &points_data[tv[0] as usize];
                                    let b = &points_data[tv[1] as usize];
                                    let c = &points_data[tv[2] as usize];
                                    let d = &points_data[tv[3] as usize];
                                    let e = &points_data[top_vert as usize];
                                    // orient3d for tet
                                    let ad = [a[0]-d[0], a[1]-d[1], a[2]-d[2]];
                                    let bd = [b[0]-d[0], b[1]-d[1], b[2]-d[2]];
                                    let cd = [c[0]-d[0], c[1]-d[1], c[2]-d[2]];
                                    let tet_orient = ad[0]*(bd[1]*cd[2]-bd[2]*cd[1]) + bd[0]*(cd[1]*ad[2]-cd[2]*ad[1]) + cd[0]*(ad[1]*bd[2]-ad[2]*bd[1]);
                                    // insphere (simplified, sign only)
                                    let ae = [a[0]-e[0], a[1]-e[1], a[2]-e[2]];
                                    let be = [b[0]-e[0], b[1]-e[1], b[2]-e[2]];
                                    let ce = [c[0]-e[0], c[1]-e[1], c[2]-e[2]];
                                    let de = [d[0]-e[0], d[1]-e[1], d[2]-e[2]];
                                    let aes = ae[0]*ae[0]+ae[1]*ae[1]+ae[2]*ae[2];
                                    let bes = be[0]*be[0]+be[1]*be[1]+be[2]*be[2];
                                    let ces = ce[0]*ce[0]+ce[1]*ce[1]+ce[2]*ce[2];
                                    let des = de[0]*de[0]+de[1]*de[1]+de[2]*de[2];
                                    let ins = ae[0]*(be[1]*(ce[2]*des-ces*de[2])-be[2]*(ce[1]*des-ces*de[1])+(bes)*(ce[1]*de[2]-ce[2]*de[1]))
                                            - ae[1]*(be[0]*(ce[2]*des-ces*de[2])-be[2]*(ce[0]*des-ces*de[0])+(bes)*(ce[0]*de[2]-ce[2]*de[0]))
                                            + ae[2]*(be[0]*(ce[1]*des-ces*de[1])-be[1]*(ce[0]*des-ces*de[0])+(bes)*(ce[0]*de[1]-ce[1]*de[0]))
                                            - aes*(be[0]*(ce[1]*de[2]-ce[2]*de[1])-be[1]*(ce[0]*de[2]-ce[2]*de[0])+be[2]*(ce[0]*de[1]-ce[1]*de[0]));
                                    let violation = (ins < 0.0 && tet_orient < 0.0) || (ins > 0.0 && tet_orient > 0.0);
                                    if !violation { continue; }
                                    // Check 2-3 orient: can top_vert go inside bot_tet?
                                    // For face opposite f, the 3 other faces include f.
                                    // We check the TetViAsSeenFrom faces
                                    let tvasf: [[usize; 3]; 4] = [[1,3,2],[0,2,3],[0,3,1],[0,1,2]];
                                    let bot_ord_vi = tvasf[f as usize];
                                    let ep = [e[0], e[1], e[2]];
                                    let mut all_ok = true;
                                    for i in 0..3 {
                                        let fv = tvasf[bot_ord_vi[i]];
                                        let p0 = &points_data[tv[fv[0]] as usize];
                                        let p1 = &points_data[tv[fv[1]] as usize];
                                        let p2 = &points_data[tv[fv[2]] as usize];
                                        let fd = [p0[0]-ep[0], p0[1]-ep[1], p0[2]-ep[2]];
                                        let gd = [p1[0]-ep[0], p1[1]-ep[1], p1[2]-ep[2]];
                                        let hd = [p2[0]-ep[0], p2[1]-ep[1], p2[2]-ep[2]];
                                        let orient = fd[0]*(gd[1]*hd[2]-gd[2]*hd[1]) + gd[0]*(hd[1]*fd[2]-hd[2]*fd[1]) + hd[0]*(fd[1]*gd[2]-fd[2]*gd[1]);
                                        if orient >= 0.0 { all_ok = false; }
                                    }
                                    eprintln!("[VIOLATION-ANALYSIS] tet[{}].f{}->tet[{}].f{}: verts={:?} top_vert={} tet_orient={:.6e} insphere={:.6e} 23_feasible={}",
                                        t, f, nei, nei_f, tv, top_vert, tet_orient, ins, all_ok);
                                    printed += 1;
                                }
                            }
                        }

                        break;
                    }

                    state.reset_flip_count(queue);
                    queue.write_buffer(&state.buffers.counters, 16, bytemuck::cast_slice(&[0u32, 0u32, 0u32, 0u32]));
                    {
                        let mut encoder = device.create_command_encoder(&Default::default());
                        encoder.copy_buffer_to_buffer(
                            &state.buffers.flip_to_tet, 0,
                            &state.buffers.flip_queue, 0,
                            (validated_flips as u64) * 4,
                        );
                        let inf_idx = state.num_points + 4;
                        state.dispatch_flip_exact(&mut encoder, queue, validated_flips, inf_idx, false, total_flips);
                        queue.submit(Some(encoder.finish()));
                        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
                    }

                    let (_next_count, metadata_flips) = state.buffers.read_flip_count(device, queue).await;

                    let debug_counters = state.buffers.read_counters(device, queue).await;
                    eprintln!(
                        "[FLIP-ALL-DBG] invalid_opp={}, dead_b={}, lock_or_dead_a={}, reval_fail={}",
                        debug_counters.scratch[0], debug_counters.scratch[1],
                        debug_counters.scratch[2], debug_counters.scratch[3]
                    );

                    if metadata_flips > 0 {
                        org_flip_nums.push((total_flips, metadata_flips));

                        let mut encoder = device.create_command_encoder(&Default::default());
                        state.dispatch_update_opp(&mut encoder, queue, total_flips, metadata_flips);
                        // No donation needed: global atomic counter, no recycling.
                        queue.submit(Some(encoder.finish()));
                        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
                    }

                    eprintln!(
                        "[FLIP-ALL] Iter {}: active={}, validated={}, executed={}",
                        flip_iter + 1, act_tet_num, validated_flips, metadata_flips
                    );

                    total_flips += metadata_flips;

                    // Update tet_num: 2-3 flips allocate new tets beyond current range
                    if metadata_flips > 0 {
                        let vfa: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.vert_free_arr, 1).await;
                        state.next_free_tet = vfa[0];
                        tet_num = state.next_free_tet.max(tet_num);
                    }

                    if metadata_flips == 0 { break; }
                    if total_flips > state.max_flips { break; }
                }

                eprintln!("[FLIP] All-tets phase complete: {} total flipped tets", total_flips);
                state.cpu_profiler.end("6_reflip_all");
            }

            // Read back next_free_tet after exact phase too (exact 2-3 flips may allocate more)
            {
                let vfa: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.vert_free_arr, 1).await;
                state.next_free_tet = vfa[0];
                eprintln!("[FLIP] next_free_tet after all flips: {}", state.next_free_tet);
            }

            // DEBUG: Post-flip adjacency check
            {
                let tet_num = state.max_tets as usize;
                let tet_info_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.tet_info, tet_num).await;
                let tet_opp_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.tet_opp, tet_num * 4).await;
                let tet_data: Vec<[u32; 4]> = state.buffers.read_buffer_as(device, queue, &state.buffers.tets, tet_num).await;
                let mut dead_refs = 0u32;
                let mut oob_refs = 0u32;
                let mut back_opp_mismatch = 0u32;
                let mut internal_count = 0u32;
                let mut geom_wrong = 0u32;
                let mut printed_mismatches = 0u32;
                let alive_count = tet_info_data.iter().filter(|&&x| (x & 1) != 0).count();
                for t in 0..tet_num {
                    if (tet_info_data[t] & 1) == 0 { continue; } // skip dead tets
                    for f in 0..4usize {
                        let opp = tet_opp_data[t * 4 + f];
                        if opp == 0xFFFFFFFF { continue; } // INVALID is ok
                        if (opp & 4) != 0 { internal_count += 1; }
                        let nei_tet = (opp >> 5) as usize;
                        let nei_face = (opp & 3) as usize;
                        if nei_tet >= tet_num {
                            oob_refs += 1;
                        } else if (tet_info_data[nei_tet] & 1) == 0 {
                            dead_refs += 1;
                        } else {
                            // Geometric check: do these tets share exactly 3 vertices on the claimed face?
                            if t < tet_data.len() && nei_tet < tet_data.len() {
                                let face_vi_map: [[usize; 3]; 4] = [[1,2,3],[0,2,3],[0,1,3],[0,1,2]];
                                let tv = tet_data[t];
                                let nv = tet_data[nei_tet];
                                let face_a: [u32; 3] = [tv[face_vi_map[f][0]], tv[face_vi_map[f][1]], tv[face_vi_map[f][2]]];
                                let face_b: [u32; 3] = [nv[face_vi_map[nei_face][0]], nv[face_vi_map[nei_face][1]], nv[face_vi_map[nei_face][2]]];
                                let shared = face_a.iter().filter(|v| face_b.contains(v)).count();
                                if shared != 3 {
                                    geom_wrong += 1;
                                    if geom_wrong <= 5 {
                                        let is_internal = (opp & 4) != 0;
                                        eprintln!("  GEOM-WRONG-FLIP: tet[{}].f{} ({:?}) -> tet[{}].f{} ({:?}), shared={}, internal={}",
                                            t, f, face_a, nei_tet, nei_face, face_b, shared, is_internal);
                                    }
                                }
                            }
                            let back_opp = tet_opp_data[nei_tet * 4 + nei_face];
                            if back_opp != 0xFFFFFFFF {
                                let back_tet = (back_opp >> 5) as usize;
                                let back_face = (back_opp & 3) as usize;
                                if back_tet != t || back_face != f {
                                    back_opp_mismatch += 1;
                                    if printed_mismatches < 5 {
                                        printed_mismatches += 1;
                                        let tv = if t < tet_data.len() { tet_data[t] } else { [0;4] };
                                        let nv = if nei_tet < tet_data.len() { tet_data[nei_tet] } else { [0;4] };
                                        let is_internal = (opp & 4) != 0;
                                        let is_changed_t = (tet_info_data[t] & 2) != 0;
                                        let is_changed_n = (tet_info_data[nei_tet] & 2) != 0;
                                        eprintln!("  MISMATCH: tet[{}].f{} (verts={:?}, changed={}) -> tet[{}].f{} (verts={:?}, changed={}, internal={})",
                                            t, f, tv, is_changed_t, nei_tet, nei_face, nv, is_changed_n, is_internal);
                                        eprintln!("    back: tet[{}].f{} -> tet[{}].f{} (opp=0x{:08x}, back=0x{:08x})",
                                            nei_tet, nei_face, back_tet, back_face, opp, back_opp);
                                        // Check shared vertices
                                        let face_vi_map: [[usize; 3]; 4] = [[1,2,3],[0,2,3],[0,1,3],[0,1,2]];
                                        let face_a: Vec<u32> = face_vi_map[f].iter().map(|&i| tv[i]).collect();
                                        let face_b: Vec<u32> = face_vi_map[nei_face].iter().map(|&i| nv[i]).collect();
                                        let shared: Vec<u32> = face_a.iter().filter(|v| face_b.contains(v)).cloned().collect();
                                        eprintln!("    face_a={:?}, face_b={:?}, shared={} verts", face_a, face_b, shared.len());
                                    }
                                }
                            }
                        }
                    }
                }
                eprintln!("[ADJACENCY-CHECK] Iter {} post-flip: alive={}, dead_refs={}, oob_refs={}, back_opp_mismatch={}, geom_wrong={}, internal={}",
                    iteration, alive_count, dead_refs, oob_refs, back_opp_mismatch, geom_wrong, internal_count);
            }

            // DEBUG: Save vert_tet BEFORE relocate for comparison
            let pre_reloc_vert_tet: Option<Vec<u32>> = if iteration == 3 {
                let num_pos = num_uninserted as usize;
                let vt: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.vert_tet, num_pos).await;
                let uninserted_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.uninserted, num_pos).await;
                let tet_data: Vec<[u32; 4]> = state.buffers.read_buffer_as(device, queue, &state.buffers.tets, state.max_tets as usize).await;
                let points_data: Vec<[f32; 4]> = state.buffers.read_buffer_as(device, queue, &state.buffers.points, state.num_points as usize + 5).await;
                let tet_info_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.tet_info, state.max_tets as usize).await;
                let inf_idx = state.num_points + 4;
                let mut ok_pr = 0u32; let mut outside_pr = 0u32; let mut dead_pr = 0u32; let mut invalid_pr = 0u32; let mut boundary_pr = 0u32;
                for pos in 0..num_pos {
                    let ti = vt[pos] as usize;
                    if ti == 0xFFFFFFFF { invalid_pr += 1; continue; }
                    if ti >= tet_data.len() { invalid_pr += 1; continue; }
                    if (tet_info_data[ti] & 1) == 0 { dead_pr += 1; continue; }
                    let verts = tet_data[ti];
                    if verts[0] >= inf_idx || verts[1] >= inf_idx || verts[2] >= inf_idx || verts[3] >= inf_idx { boundary_pr += 1; continue; }
                    let vertex_id = uninserted_data[pos];
                    let p = [points_data[vertex_id as usize][0], points_data[vertex_id as usize][1], points_data[vertex_id as usize][2]];
                    let tv: Vec<[f32; 3]> = (0..4).map(|i| [points_data[verts[i] as usize][0], points_data[verts[i] as usize][1], points_data[verts[i] as usize][2]]).collect();
                    let faces = [[1,3,2],[0,2,3],[0,3,1],[0,1,2]];
                    let tet_orient = {
                        let ad = [tv[0][0]-tv[3][0], tv[0][1]-tv[3][1], tv[0][2]-tv[3][2]];
                        let bd = [tv[1][0]-tv[3][0], tv[1][1]-tv[3][1], tv[1][2]-tv[3][2]];
                        let cd = [tv[2][0]-tv[3][0], tv[2][1]-tv[3][1], tv[2][2]-tv[3][2]];
                        ad[0]*(bd[1]*cd[2]-bd[2]*cd[1]) + bd[0]*(cd[1]*ad[2]-cd[2]*ad[1]) + cd[0]*(ad[1]*bd[2]-ad[2]*bd[1])
                    };
                    let mut is_inside = true;
                    for face in &faces {
                        let a = tv[face[0]]; let b = tv[face[1]]; let c = tv[face[2]];
                        let ad = [a[0]-p[0], a[1]-p[1], a[2]-p[2]];
                        let bd = [b[0]-p[0], b[1]-p[1], b[2]-p[2]];
                        let cd = [c[0]-p[0], c[1]-p[1], c[2]-p[2]];
                        let det = ad[0]*(bd[1]*cd[2]-bd[2]*cd[1]) + bd[0]*(cd[1]*ad[2]-cd[2]*ad[1]) + cd[0]*(ad[1]*bd[2]-ad[2]*bd[1]);
                        if tet_orient < 0.0 && det > 0.0 { is_inside = false; }
                        if tet_orient > 0.0 && det < 0.0 { is_inside = false; }
                    }
                    if is_inside { ok_pr += 1; } else { outside_pr += 1; }
                }
                eprintln!("[VERT_TET_PRE_RELOC] Iter {}: {} inside, {} OUTSIDE, {} dead, {} invalid, {} boundary (of {} uninserted)",
                    iteration, ok_pr, outside_pr, dead_pr, invalid_pr, boundary_pr, num_pos);
                Some(vt)
            } else { None };

            // ========== RELOCATE ALL ==========
            // Port of GpuDelaunay.cu:1250-1311 (relocateAll)
            // After flipping, update vert_tet for all uninserted vertices by tracing
            // the flip chain. Without this, vert_tet becomes stale after flips reuse
            // tet slots, causing vertices to vote for wrong tets in the next iteration.
            if total_flips > 0 && !org_flip_nums.is_empty() {
                state.cpu_profiler.begin("7_relocate_all");

                // Step 1: Initialize tet_to_flip to -1
                // CRITICAL: Must cover ALL possible tet indices, not just current_tet_num.
                // Flip items reference tet slots allocated from free lists which can have
                // indices beyond current_tet_num. Stale entries cause infinite loops!
                {
                    let max_tet_num = state.max_tets;
                    let mut encoder = device.create_command_encoder(&Default::default());
                    state.dispatch_init_tet_to_flip(&mut encoder, queue, max_tet_num);
                    queue.submit(Some(encoder.finish()));
                    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
                }

                // Step 2: Build flip trace in REVERSE order (CUDA: GpuDelaunay.cu:1269-1282)
                // Later flips must be processed first so that the chain starts from
                // the most recent flip touching each tet slot.
                for &(org_flip_num, flip_num) in org_flip_nums.iter().rev() {
                    let mut encoder = device.create_command_encoder(&Default::default());
                    state.dispatch_update_flip_trace(&mut encoder, queue, org_flip_num, flip_num);
                    queue.submit(Some(encoder.finish()));
                    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
                }

                // Step 3: Relocate points by following flip chain
                {
                    let mut encoder = device.create_command_encoder(&Default::default());
                    state.dispatch_relocate_points_fast(&mut encoder, queue, num_uninserted, total_flips);
                    queue.submit(Some(encoder.finish()));
                    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
                }

                eprintln!("[RELOCATE] Updated vert_tet via {} flip batches ({} total flips)", org_flip_nums.len(), total_flips);

                // DEBUG: Check vert_tet after relocate (before compaction)
                if iteration >= 2 && iteration <= 4 {
                    let num_pos = num_uninserted as usize;
                    let vert_tet_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.vert_tet, num_pos).await;
                    let uninserted_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.uninserted, num_pos).await;
                    let tet_data: Vec<[u32; 4]> = state.buffers.read_buffer_as(device, queue, &state.buffers.tets, state.max_tets as usize).await;
                    let points_data: Vec<[f32; 4]> = state.buffers.read_buffer_as(device, queue, &state.buffers.points, state.num_points as usize + 5).await;
                    let tet_info_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.tet_info, state.max_tets as usize).await;
                    // Also read tet_to_flip for detailed analysis
                    let tet_to_flip_data: Vec<i32> = state.buffers.read_buffer_as(device, queue, &state.buffers.tet_to_flip, state.max_tets as usize).await;
                    let inf_idx = state.num_points + 4;
                    let mut ok_r = 0u32; let mut outside_r = 0u32; let mut dead_r = 0u32; let mut invalid_r = 0u32; let mut boundary_r = 0u32;
                    let mut outside_details: Vec<String> = Vec::new();
                    for pos in 0..num_pos {
                        let ti = vert_tet_data[pos] as usize;
                        if ti == 0xFFFFFFFF { invalid_r += 1; continue; }
                        if ti >= tet_data.len() { invalid_r += 1; continue; }
                        if (tet_info_data[ti] & 1) == 0 { dead_r += 1; continue; }
                        let verts = tet_data[ti];
                        if verts[0] >= inf_idx || verts[1] >= inf_idx || verts[2] >= inf_idx || verts[3] >= inf_idx { boundary_r += 1; continue; }
                        let vertex_id = uninserted_data[pos];
                        let p = [points_data[vertex_id as usize][0], points_data[vertex_id as usize][1], points_data[vertex_id as usize][2]];
                        let tv: Vec<[f32; 3]> = (0..4).map(|i| [points_data[verts[i] as usize][0], points_data[verts[i] as usize][1], points_data[verts[i] as usize][2]]).collect();
                        let faces = [[1,3,2],[0,2,3],[0,3,1],[0,1,2]];
                        let tet_orient = {
                            let ad = [tv[0][0]-tv[3][0], tv[0][1]-tv[3][1], tv[0][2]-tv[3][2]];
                            let bd = [tv[1][0]-tv[3][0], tv[1][1]-tv[3][1], tv[1][2]-tv[3][2]];
                            let cd = [tv[2][0]-tv[3][0], tv[2][1]-tv[3][1], tv[2][2]-tv[3][2]];
                            ad[0]*(bd[1]*cd[2]-bd[2]*cd[1]) + bd[0]*(cd[1]*ad[2]-cd[2]*ad[1]) + cd[0]*(ad[1]*bd[2]-ad[2]*bd[1])
                        };
                        let mut is_inside = true;
                        for face in &faces {
                            let a = tv[face[0]]; let b = tv[face[1]]; let c = tv[face[2]];
                            let ad = [a[0]-p[0], a[1]-p[1], a[2]-p[2]];
                            let bd = [b[0]-p[0], b[1]-p[1], b[2]-p[2]];
                            let cd = [c[0]-p[0], c[1]-p[1], c[2]-p[2]];
                            let det = ad[0]*(bd[1]*cd[2]-bd[2]*cd[1]) + bd[0]*(cd[1]*ad[2]-cd[2]*ad[1]) + cd[0]*(ad[1]*bd[2]-ad[2]*bd[1]);
                            if tet_orient < 0.0 && det > 0.0 { is_inside = false; }
                            if tet_orient > 0.0 && det < 0.0 { is_inside = false; }
                        }
                        if is_inside {
                            ok_r += 1;
                        } else {
                            outside_r += 1;
                            // Detailed analysis for OUTSIDE entries in iter 3
                            if iteration == 3 && outside_details.len() < 10 {
                                let pre_ti = pre_reloc_vert_tet.as_ref().map(|v| v[pos]).unwrap_or(0xDEAD);
                                let t2f = if ti < tet_to_flip_data.len() { tet_to_flip_data[ti] } else { -999 };
                                let pre_t2f = if (pre_ti as usize) < tet_to_flip_data.len() { tet_to_flip_data[pre_ti as usize] } else { -999 };
                                let changed = pre_ti != vert_tet_data[pos];
                                outside_details.push(format!(
                                    "pos={} vert={} pre_tet={} post_tet={} changed={} t2f[post]={} t2f[pre]={}",
                                    pos, vertex_id, pre_ti, ti, changed, t2f, pre_t2f));
                            }
                        }
                    }
                    eprintln!("[VERT_TET_POST_RELOC] Iter {}: {} inside, {} OUTSIDE, {} dead, {} invalid, {} boundary (of {} uninserted)",
                        iteration, ok_r, outside_r, dead_r, invalid_r, boundary_r, num_pos);
                    if !outside_details.is_empty() {
                        for detail in &outside_details {
                            eprintln!("[RELOC-OUTSIDE] {}", detail);
                        }
                    }

                    // CPU-side chain trace for first OUTSIDE entry in iter 3
                    if iteration == 3 && outside_r > 0 {
                        if let Some(ref pre_vt) = pre_reloc_vert_tet {
                            let flip_arr_data: Vec<[i32; 4]> = state.buffers.read_buffer_as(device, queue, &state.buffers.flip_arr, (total_flips as usize) * 2).await;
                            // Find first OUTSIDE entry
                            for pos in 0..num_pos {
                                let ti_post = vert_tet_data[pos] as usize;
                                if ti_post == 0xFFFFFFFF || ti_post >= tet_data.len() { continue; }
                                if (tet_info_data[ti_post] & 1) == 0 { continue; }
                                let verts = tet_data[ti_post];
                                if verts[0] >= inf_idx || verts[1] >= inf_idx || verts[2] >= inf_idx || verts[3] >= inf_idx { continue; }
                                let vertex_id = uninserted_data[pos];
                                let p = [points_data[vertex_id as usize][0], points_data[vertex_id as usize][1], points_data[vertex_id as usize][2]];
                                let tv: Vec<[f32; 3]> = (0..4).map(|i| [points_data[verts[i] as usize][0], points_data[verts[i] as usize][1], points_data[verts[i] as usize][2]]).collect();
                                let faces = [[1,3,2],[0,2,3],[0,3,1],[0,1,2]];
                                let tet_orient = {
                                    let ad = [tv[0][0]-tv[3][0], tv[0][1]-tv[3][1], tv[0][2]-tv[3][2]];
                                    let bd = [tv[1][0]-tv[3][0], tv[1][1]-tv[3][1], tv[1][2]-tv[3][2]];
                                    let cd = [tv[2][0]-tv[3][0], tv[2][1]-tv[3][1], tv[2][2]-tv[3][2]];
                                    ad[0]*(bd[1]*cd[2]-bd[2]*cd[1]) + bd[0]*(cd[1]*ad[2]-cd[2]*ad[1]) + cd[0]*(ad[1]*bd[2]-ad[2]*bd[1])
                                };
                                let mut is_outside = false;
                                for face in &faces {
                                    let a = tv[face[0]]; let b = tv[face[1]]; let c = tv[face[2]];
                                    let ad = [a[0]-p[0], a[1]-p[1], a[2]-p[2]];
                                    let bd = [b[0]-p[0], b[1]-p[1], b[2]-p[2]];
                                    let cd = [c[0]-p[0], c[1]-p[1], c[2]-p[2]];
                                    let det = ad[0]*(bd[1]*cd[2]-bd[2]*cd[1]) + bd[0]*(cd[1]*ad[2]-cd[2]*ad[1]) + cd[0]*(ad[1]*bd[2]-ad[2]*bd[1]);
                                    if tet_orient < 0.0 && det > 0.0 { is_outside = true; }
                                    if tet_orient > 0.0 && det < 0.0 { is_outside = true; }
                                }
                                if !is_outside { continue; } // Only trace OUTSIDE entries
                                let ti_pre = pre_vt[pos] as usize;
                                eprintln!("[CHAIN-TRACE] pos={} vert={} pre_tet={} post_tet={}", pos, vertex_id, ti_pre, ti_post);
                                eprintln!("  post_tet[{}] verts={:?}", ti_post, tet_data[ti_post]);
                                eprintln!("  point coords=({:.6}, {:.6}, {:.6})", p[0], p[1], p[2]);
                                // Trace the chain from pre_tet
                                let mut chain_tet = ti_pre;
                                let mut chain_next = tet_to_flip_data[chain_tet];
                                eprintln!("  tet_to_flip[{}] = {} (flag={}, dest={})", chain_tet, chain_next, chain_next & 1, chain_next >> 1);
                                if chain_next != -1 {
                                    let mut flag = chain_next & 1;
                                    let mut dest = chain_next >> 1;
                                    let mut steps = 0;
                                    while flag == 1 && steps < 5 {
                                        steps += 1;
                                        let fi = dest as usize;
                                        if fi * 2 + 1 >= flip_arr_data.len() { eprintln!("  [CHAIN] flip {} out of range", fi); break; }
                                        let fv0 = flip_arr_data[fi * 2];
                                        let fv1 = flip_arr_data[fi * 2 + 1];
                                        let v = [fv0[0] as u32, fv0[1] as u32, fv0[2] as u32, fv0[3] as u32, fv1[0] as u32];
                                        let t = [fv1[1], fv1[2], fv1[3]];
                                        let f_type = if t[2] < 0 { "Flip32" } else { "Flip23" };
                                        eprintln!("  [CHAIN] step {} flip[{}] type={} v={:?} t={:?}", steps, fi, f_type, v, t);
                                        // Compute orient3d tests
                                        let load_pt = |vid: u32| -> [f32; 3] {
                                            let vi = vid as usize;
                                            if vi < points_data.len() {
                                                [points_data[vi][0], points_data[vi][1], points_data[vi][2]]
                                            } else {
                                                [0.0, 0.0, 0.0]
                                            }
                                        };
                                        let pv: Vec<[f32; 3]> = v.iter().map(|&vi| load_pt(vi)).collect();
                                        // Test 1
                                        let (fa, fb, fc) = if t[2] >= 0 { (0usize, 2, 3) } else { (0, 1, 2) };
                                        let orient_fn = |a: [f32;3], b: [f32;3], c: [f32;3], d: [f32;3]| -> f32 {
                                            let ad = [a[0]-d[0], a[1]-d[1], a[2]-d[2]];
                                            let bd = [b[0]-d[0], b[1]-d[1], b[2]-d[2]];
                                            let cd = [c[0]-d[0], c[1]-d[1], c[2]-d[2]];
                                            ad[0]*(bd[1]*cd[2]-bd[2]*cd[1]) + bd[0]*(cd[1]*ad[2]-cd[2]*ad[1]) + cd[0]*(ad[1]*bd[2]-ad[2]*bd[1])
                                        };
                                        let has_inf_fn = |va: u32, vb: u32, vc: u32| -> bool {
                                            va >= inf_idx as u32 || vb >= inf_idx as u32 || vc >= inf_idx as u32
                                        };
                                        let mut det1 = orient_fn(pv[fa], pv[fb], pv[fc], p);
                                        let has_inf1 = has_inf_fn(v[fa], v[fb], v[fc]);
                                        if has_inf1 { det1 = -det1; }
                                        let ord0 = if det1 < 0.0 { 1i32 } else if det1 > 0.0 { -1 } else { 0 };
                                        eprintln!("    test1: orient3d(v[{}],v[{}],v[{}],pt) det={:.6e} inf={} ord={}", fa, fb, fc, det1, has_inf1, ord0);
                                        let next_loc_id;
                                        if t[2] < 0 {
                                            // Flip32
                                            next_loc_id = if ord0 == 1 { 0 } else { 1 };
                                        } else {
                                            // Flip23
                                            #[allow(unused_variables)]
                                            let nlid_base: i32;
                                            if ord0 == 1 {
                                                nlid_base = 0i32;
                                                // (0, 3, 1)
                                                let mut det2 = orient_fn(pv[0], pv[3], pv[1], p);
                                                if has_inf_fn(v[0], v[3], v[1]) { det2 = -det2; }
                                                let ord1 = if det2 < 0.0 { 1i32 } else if det2 > 0.0 { -1 } else { 0 };
                                                eprintln!("    test2: orient3d(v[0],v[3],v[1],pt) det={:.6e} ord={}", det2, ord1);
                                                next_loc_id = if ord1 != 1 { 2 } else { nlid_base };
                                            } else {
                                                nlid_base = 1i32;
                                                // (0, 4, 3)
                                                let mut det2 = orient_fn(pv[0], pv[4], pv[3], p);
                                                if has_inf_fn(v[0], v[4], v[3]) { det2 = -det2; }
                                                let ord1 = if det2 < 0.0 { 1i32 } else if det2 > 0.0 { -1 } else { 0 };
                                                eprintln!("    test2: orient3d(v[0],v[4],v[3],pt) det={:.6e} ord={}", det2, ord1);
                                                next_loc_id = if ord1 != 1 { 2 } else { nlid_base };
                                            }
                                        }
                                        eprintln!("    -> next_loc_id={} t[{}]={}", next_loc_id, next_loc_id, t[next_loc_id as usize]);
                                        let ni = t[next_loc_id as usize];
                                        flag = ni & 1;
                                        dest = ni >> 1;
                                        if flag == 0 {
                                            eprintln!("    -> TERMINAL: tet {}", dest);
                                            // Check containment in terminal tet
                                            let final_tet = dest as usize;
                                            if final_tet < tet_data.len() {
                                                let fv = tet_data[final_tet];
                                                eprintln!("    terminal tet[{}] verts={:?}", final_tet, fv);
                                            }
                                        }
                                    }
                                }
                                // Only trace first OUTSIDE entry
                                break;
                            }
                        }
                    }
                }

                state.cpu_profiler.end("7_relocate_all");
            }

            // DEBUG: Orient check after flips + relocate
            {
                let tet_num = state.current_tet_num as usize;
                let points_data: Vec<[f32; 4]> = state.buffers.read_buffer_as(device, queue, &state.buffers.points, state.num_points as usize + 5).await;
                let tet_data: Vec<[u32; 4]> = state.buffers.read_buffer_as(device, queue, &state.buffers.tets, tet_num).await;
                let tet_info_data: Vec<u32> = state.buffers.read_buffer_as(device, queue, &state.buffers.tet_info, tet_num).await;
                let inf_idx = state.num_points + 4;
                let mut pos_count = 0u32;
                let mut neg_count = 0u32;
                for t in 0..tet_num {
                    if (tet_info_data[t] & 1) == 0 { continue; }
                    let v = tet_data[t];
                    if v[0] >= inf_idx || v[1] >= inf_idx || v[2] >= inf_idx || v[3] >= inf_idx { continue; }
                    let a = [points_data[v[0] as usize][0], points_data[v[0] as usize][1], points_data[v[0] as usize][2]];
                    let b = [points_data[v[1] as usize][0], points_data[v[1] as usize][1], points_data[v[1] as usize][2]];
                    let c = [points_data[v[2] as usize][0], points_data[v[2] as usize][1], points_data[v[2] as usize][2]];
                    let d = [points_data[v[3] as usize][0], points_data[v[3] as usize][1], points_data[v[3] as usize][2]];
                    let ad = [a[0]-d[0], a[1]-d[1], a[2]-d[2]];
                    let bd = [b[0]-d[0], b[1]-d[1], b[2]-d[2]];
                    let cd = [c[0]-d[0], c[1]-d[1], c[2]-d[2]];
                    let det = ad[0]*(bd[1]*cd[2]-bd[2]*cd[1]) + bd[0]*(cd[1]*ad[2]-cd[2]*ad[1]) + cd[0]*(ad[1]*bd[2]-ad[2]*bd[1]);
                    if det > 0.0 { pos_count += 1; } else if det < 0.0 { neg_count += 1; }
                }
                eprintln!("[ORIENT-POST-FLIP] Iter {}: {} positive, {} negative orient3d (AFTER flips+relocate)", iteration, pos_count, neg_count);
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
