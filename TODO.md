# TODO: Path to 66/66 Tests

## Critical Path: Fix the Cascade

### Priority 1: Debug 4-Tet Allocation (BLOCKER)

**Goal:** Make `get_free_slots_4tet` work without insertion failures

**Current Blocker:** Unknown bug causes "3 out of 4 points fail to insert"

**Debugging Steps:**
1. **Add instrumentation:**
   ```wgsl
   // Counter-based debugging
   if (conditions_met) {
       atomicAdd(&counters[COUNTER_DEBUG_1], 1u);
   }
   ```
   Track how many splits succeed vs fail

2. **Verify each step:**
   - [ ] Check `free_arr` contains expected values
   - [ ] Check `vert_free_arr` decrement works
   - [ ] Check new tet indices are written correctly
   - [ ] Check old tet is marked dead at right time
   - [ ] Check flip queue gets correct indices

3. **Test variable naming hypothesis:**
   - Try keeping `t0` name but allocating from free_arr
   - Check if code elsewhere assumes `t0 == insert_list[idx].x`

4. **Test minimal changes:**
   - Allocate 4 tets but DON'T mark old tet dead
   - Mark old tet dead but DON'T allocate 4 (use 3)
   - Isolate which change breaks it

5. **Compare with CUDA:**
   - Re-read CUDA's 4-tet allocation carefully
   - Check if we're missing initialization steps
   - Verify donation logic isn't required for basic function

**Success Criteria:** 4-point test passes with 4-tet allocation

---

### Priority 2: One-Pass Split (DEPENDS ON #1)

**Goal:** Remove color batching, split all tets simultaneously

**Prerequisites:** 4-tet allocation working

**Changes:**
```rust
// src/phase1.rs
// Remove: compute_and_upload_coloring
// Remove: for color in 0..num_colors loop
// Change to single dispatch:
state.dispatch_split(&mut encoder, queue, num_inserted, 0);
```

```wgsl
// src/shaders/split.wgsl
// Remove color filtering:
// let my_color = tet_color[t0];
// if my_color != color { return; }
```

**Success Criteria:** Tests don't regress (stay at 51/66 minimum)

---

### Priority 3: Concurrent Split Detection (DEPENDS ON #2)

**Goal:** Detect when neighbor is also splitting, find correct new tet

**Prerequisites:** One-pass split working, 4-tet allocation working

**Implementation:**
```wgsl
let nei_split_vert = tet_to_vert[nei_tet];

if nei_split_vert != INVALID {
    // Neighbor is splitting - find which new tet has this face
    let nei_block_top = (nei_split_vert + 1u) * MEAN_VERTEX_DEGREE - 1u;
    nei_tet = free_arr[nei_block_top - nei_face];  // CUDA formula
    nei_face = 3u;  // Face opposite P in new tet
}
```

**Testing:**
- Start with simple meshes (verify no regression)
- Test with meshes where concurrent splits occur
- Verify formula finds correct tet

**Success Criteria:** Tests stay stable or improve (51-58/66)

---

### Priority 4: Bidirectional Updates (DEPENDS ON #3)

**Goal:** Update neighbor's back-pointer safely

**Prerequisites:** Concurrent split detection working

**Implementation:**
```wgsl
// Set our pointer
set_opp_at(my_tet, k, encode_opp(nei_tet, nei_face));

// Set neighbor's back-pointer (now safe!)
set_opp_at(nei_tet, nei_face, encode_opp(my_tet, k));
```

**Testing:**
- Run full test suite
- Check Euler characteristic on large meshes
- Verify adjacency is bidirectional

**Success Criteria:** 66/66 tests passing ✅

---

## Alternative Path: Two-Phase Adjacency Update

If 4-tet allocation proves too difficult, implement two-phase update:

### Phase 1: Collect Updates
```wgsl
// New buffer: adjacency_updates[max_tets * 4]
// Each entry: packed (from_tet, from_face, to_tet, to_face)

for each splitting tet {
    for each external face {
        let nei_tet = decode_opp_tet(opp);
        let nei_face = decode_opp_face(opp);

        // Write update to buffer
        let update_idx = atomicAdd(&update_count, 1u);
        adjacency_updates[update_idx] = pack(my_tet, my_face, nei_tet, nei_face);
    }
}
```

### Phase 2: Apply Updates
```wgsl
// Separate kernel dispatch
for each update {
    let (from_tet, from_face, to_tet, to_face) = unpack(update);
    set_opp_at(to_tet, to_face, encode_opp(from_tet, from_face));
}
```

**Pros:**
- Eliminates all race conditions
- Simpler than concurrent split detection
- Doesn't require 4-tet allocation

**Cons:**
- Extra buffer (4 * max_tets * sizeof(u32))
- Extra kernel dispatch (slower)
- Not faithful to CUDA approach

**Success Criteria:** 66/66 tests, but with performance cost

---

## Recover Lost Work

The following improvements were made this session but lost in git reset:

### Graph Coloring (58/66 state)
1. Create `src/coloring.rs`:
   ```rust
   pub fn color_tets(
       tet_indices: &[u32],
       tet_adjacency: &[Vec<u32>],
       max_colors: u32,
   ) -> Vec<u32> {
       // Greedy graph coloring
   }
   ```

2. Update `src/phase1.rs`:
   ```rust
   async fn compute_and_upload_coloring(...) -> u32 {
       // Read back insert_list and tet_opp
       // Build adjacency graph
       // Compute coloring
       // Upload to GPU
   }
   ```

3. Update `src/gpu/buffers.rs`:
   - Add `tet_color: wgpu::Buffer`

4. Update `src/gpu/pipelines.rs`:
   - Add tet_color bindings to split pipeline

5. Update `src/shaders/split.wgsl`:
   - Add color filtering based on `tet_color[t0]`

**Result:** Improved from 51/66 → 58/66 tests

---

## Long-Term Improvements

### After 66/66 Achieved

1. **Performance Optimization:**
   - Profile GPU kernels
   - Optimize memory access patterns
   - Reduce atomic operations

2. **CUDA Donation System:**
   - Implement `insVertVec` mapping
   - Port donation logic fully
   - Enable block refilling

3. **Better Error Handling:**
   - Detect and report allocation failures
   - Graceful degradation for large meshes
   - User-facing error messages

4. **Testing:**
   - Add more edge cases
   - Stress test with 10k+ points
   - Performance benchmarks

---

## Notes

- Keep STATUS.md updated with progress
- Document any new bugs in CUDA_DIFFERENCES.md
- Update this TODO as priorities change
- Commit incremental progress (don't lose work again!)
