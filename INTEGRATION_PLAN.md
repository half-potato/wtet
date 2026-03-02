# Integration Plan for 14 New Kernels

## Current Status
- ✅ 14 shader files created and ready
- ✅ Struct fields added to `Pipelines`
- ⬜ Missing buffers need to be added to `GpuBuffers`
- ⬜ Pipeline creation code needs ~600 lines added
- ⬜ Dispatch functions need to be added
- ⬜ Call sites need to be wired up

## Missing Buffers (Need to Add to GpuBuffers)

### For Flip Kernels
1. **flip_arr**: `FlipItem` array (stored as vec4<i32> pairs)
   - Size: `max_tets * 2 * sizeof(vec4<i32>)`
   - Used by: update_opp, update_flip_trace, mark_rejected_flips

2. **tet_msg_arr**: `vec2<i32>` messages for concurrent flip detection
   - Size: `max_tets * sizeof(vec2<i32>)`
   - Used by: update_opp

3. **encoded_face_vi_arr**: `i32` encoded face vertex indices
   - Size: `max_tets * sizeof(i32)`
   - Used by: update_opp

4. **tet_to_flip**: `i32` flip trace chain
   - Size: `max_tets * sizeof(i32)`
   - Used by: update_flip_trace

### For Compaction/Reordering Kernels
5. **scatter_arr**: `u32` scatter map for reordering
   - Size: `(num_points + 4) * sizeof(u32)`
   - Used by: make_reverse_map, update_block_vert_free_list

6. **order_arr**: `u32` reordering map
   - Size: `num_insertions * sizeof(u32)`
   - Used by: update_block_opp_tet_idx, update_tet_idx

7. **ins_vert_vec**: `u32` inserted vertex array
   - Size: `num_insertions * sizeof(u32)`
   - Used by: make_reverse_map, update_block_vert_free_list

8. **rev_map_arr**: `u32` reverse mapping
   - Size: `num_insertions * sizeof(u32)`
   - Used by: make_reverse_map

### Note on Existing Buffers
- `free_arr` ✅ exists
- `vert_free_arr` ✅ exists
- `prefix_sum_data` ✅ exists (use as `prefix_sum`)

## Two Approaches

### Approach A: Add All Buffers First (Systematic)
1. Add 8 missing buffer fields to `GpuBuffers` struct
2. Initialize them in `GpuBuffers::new()`
3. Add all 14 pipeline creations (~600 lines)
4. Add all 14 dispatch functions (~200 lines)
5. Wire into phase1.rs based on CUDA flow
6. Run tests and debug

**Pros**: Complete, faithful to CUDA from the start
**Cons**: Large upfront work before any testing

### Approach B: Incremental with Stubs (Fast Feedback)
1. Add pipeline struct fields (✅ done)
2. Create stub pipeline implementations that compile
3. Run tests to see current failures
4. Add missing buffers/pipelines based on test failures
5. Iterate until all kernels integrated

**Pros**: Faster feedback, guided by actual failures
**Cons**: May introduce architectural differences

## Recommendation

Given the complexity and your preference for Option 1 (faithful integration), I recommend **Approach A**:

1. I'll add all 8 missing buffers to `GpuBuffers` (1 file edit)
2. I'll add all 14 pipeline creations (1 large file edit to `pipelines.rs`)
3. I'll add all 14 dispatch functions (1 file edit to `dispatch.rs`)
4. Then you can decide call site integration based on CUDA flow

This gets us to a compilable state with all infrastructure in place, ready for debugging.

**Should I proceed with Approach A?**
