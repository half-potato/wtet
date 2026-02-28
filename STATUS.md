# Implementation Status

**Last Updated:** 2026-02-26
**Test Results:** 51/66 passing (77%)

## Working Features

✅ **Core Delaunay Triangulation**
- GPU-accelerated 3D point insertion
- Stochastic point location (walk-based)
- 1-to-4 tetrahedral splits
- Delaunay flip checking
- CPU fallback (star splaying) for failed insertions

✅ **Block-Based Allocation**
- Per-vertex tet block allocation (MEAN_VERTEX_DEGREE = 64)
- 3-tet allocation with reuse (allocates 3 new, reuses original)
- Flat atomic tet_opp array for concurrent access

✅ **Most Test Cases**
- Small meshes (4-20 points): 100% pass
- Medium meshes (50-100 points): ~90% pass
- Structured grids: Most pass
- Special cases (coplanar, cospherical, duplicates): All pass

## Known Issues

### Issue #1: 10-15 Test Failures (Large Meshes)
**Affected Tests:**
- `test_raw_uniform_100` through `test_raw_uniform_1000`
- `test_raw_clustered_100`
- `test_raw_sphere_shell_200`
- `test_delaunay_uniform_300`
- Others with 100+ points

**Symptoms:**
- Euler characteristic violations (+1 to +8, should be 0 or 1)
- Broken adjacency back-pointers
- Only affects large meshes

**Root Cause:** One-way adjacency updates (see Issue #2)

### Issue #2: Broken Bidirectional Adjacency Updates
**Status:** Attempted multiple approaches, all failed

**The Problem:**
Our split kernel only updates ONE direction of adjacency:
```wgsl
// We set our pointer to neighbor
set_opp_at(my_tet, k, encode_opp(nei_tet, nei_face));

// But DON'T update neighbor's back-pointer to us
// set_opp_at(nei_tet, nei_face, encode_opp(my_tet, k));  // ← Missing!
```

This leaves broken back-pointers that accumulate in large meshes.

**Why We Can't Fix It:** See Issue #3 (cascade dependency)

### Issue #3: Shared-Neighbor Race Condition
**Status:** Identified but not solved

**The Problem:**
Multiple splitting tets can share a common neighbor:
```
Tet A (color 0) ←→ Tet C (color 1)  [face 0]
Tet B (color 0) ←→ Tet C (color 1)  [face 2]

When A and B split in parallel:
- Thread 1: set_opp_at(C, face_0, ...)  ← Updates C's face 0
- Thread 2: set_opp_at(C, face_2, ...)  ← Updates C's face 2
```

Even though they update different faces, empirical testing shows this corrupts adjacency (52/66 tests, down from 58/66).

**CUDA's Solution:** Concurrent split detection (see Issue #4)

### Issue #4: Failed Concurrent Split Detection
**Status:** Multiple attempts, all failed

**The Problem:**
CUDA handles shared neighbors by detecting when a neighbor is ALSO splitting:
```cuda
if (tetToVert[nei_tet] == INT_MAX) {
    // Neighbor NOT splitting → safe to update
    oppArr[nei_tet].setOpp(...);
} else {
    // Neighbor IS splitting → find which NEW tet to update
    nei_tet = freeArr[(nei_vertex + 1) * MeanVertDegree - 1 - nei_face];
    // Update NEW tet instead of old one
}
```

**Our Attempts:**
1. **4-color batching + concurrent detection:** 55/66 tests
   - Issue: Sequential color passes make `tet_to_vert` stale
   - When color 1 checks neighbors, color 0 already finished

2. **One-pass + concurrent detection:** 54/66 tests
   - Issue: Requires 4-tet allocation for formula to work
   - See Issue #5

**Why It Needs 4-Tet:** Formula `freeArr[block_top - face]` assumes:
- All 4 new tets allocated from `free_arr[block_top]` downwards
- Tet with external face `k` is at `free_arr[block_top - k]`

Our 3-tet allocation (reusing t0) breaks this mapping.

### Issue #5: 4-Tet Allocation Fails
**Status:** BLOCKED - Unknown bug prevents basic functionality

**The Problem:**
Attempting to allocate 4 new tets (without reusing original) causes massive failures:
```wgsl
// Allocate 4 new tets
let slots = get_free_slots_4tet(vertex);
let t0 = slots.x;  // All from free_arr
let t1 = slots.y;
let t2 = slots.z;
let t3 = slots.w;

// Mark old tet as dead
tet_info[old_tet] = 0u;
```

**Result:** "3 out of 4 points fail to insert" even in trivial 4-point test

**Debugging Findings:**
- ✅ Tet indices from `free_arr` are valid ([64, 63, 62, 61] for vertex 0)
- ✅ Bounds checking doesn't trigger (values < max_tets)
- ✅ 3-tet allocation works fine (tested)
- ❌ Unknown issue prevents insertions

**Suspected Causes:**
1. Using `old_tet` variable instead of `t0` breaks assumptions elsewhere
2. Marking old tet dead before/after writing new tets causes timing issue
3. Some other part of pipeline expects `t0` to be from `insert_list`
4. Buffer initialization bug (though `vert_free_arr` bug was found and fixed)

**Bug Found During Investigation:**
```rust
// WRONG: All vertices get MEAN_VERTEX_DEGREE
let vert_free_data = vec![MEAN_VERTEX_DEGREE; num_vertices];

// RIGHT: Count actual tets per vertex
let mut vert_free_data = vec![0u32; num_vertices];
for vertex in 0..num_vertices {
    let mut count = 0u32;
    for slot in block_range {
        if tet_idx < max_tets {
            free_data[slot] = tet_idx;
            tet_idx += 1;
            count += 1;
        }
    }
    vert_free_data[vertex] = count;  // Actual count
}
```

But fixing this didn't resolve the insertion failures.

## The Dependency Cascade

```
Broken back-pointers (10-15 test failures)
    ↓ requires
Bidirectional adjacency updates
    ↓ requires
Concurrent split detection (to avoid shared-neighbor race)
    ↓ requires
Formula: freeArr[block_top - face]
    ↓ requires
4-tet allocation (all tets from free_arr)
    ↓ BLOCKED
Unknown bug prevents 4-tet allocation from working
```

**We are blocked at the bottom of the cascade.**

## Attempts Made This Session

### 1. Proper Graph Coloring (Successful Improvement)
- Implemented CPU-side greedy graph coloring
- Ensures no two adjacent tets have same color
- Result: Improved from 51 → 58 tests
- **Status:** Working but uncommitted (lost in git reset)

### 2. Bidirectional Updates with Coloring (Failed)
- Enabled bidirectional updates with graph coloring
- Result: 52/66 tests (worse than 58)
- Issue: Shared neighbors still cause corruption

### 3. One-Pass Split (Failed)
- Removed 4-color batching, process all splits simultaneously
- Added concurrent split detection
- Result: 54-55/66 tests
- Issues: Requires 4-tet allocation, which is broken

### 4. 4-Tet Allocation (Failed - Current Blocker)
- Implemented CUDA-style 4-tet allocation
- Fixed `vert_free_arr` initialization bug
- Result: Massive insertion failures (3/4 points fail)
- **Status:** Unresolved

## What Would Fix Everything

If 4-tet allocation worked:
1. ✅ Enable one-pass splitting (no color batching)
2. ✅ Enable concurrent split detection (formula works)
3. ✅ Enable bidirectional updates (safe with detection)
4. ✅ Fix all 10-15 test failures (correct adjacency)
5. ✅ Match CUDA implementation exactly

**Estimated impact:** 51/66 → 66/66 tests (100%)

## Alternative Approaches (Not Pursued)

### Two-Phase Adjacency Update
```rust
// Phase 1: Collect updates
for each splitting tet {
    read neighbors
    write updates to temp buffer  // No races - separate buffer
}

// Phase 2: Apply updates
for each update in buffer {
    write to tet_opp  // No races - no reads
}
```

**Pros:** Eliminates all race conditions
**Cons:** Extra memory, extra dispatch, slower

### Accept Current State
**Pros:** 77% tests passing is usable
**Cons:** Large meshes unreliable, not faithful to CUDA

## Recommendations

### Short Term
1. **Commit the graph coloring work** (58/66 state before it's lost)
2. **Document 4-tet allocation bug** for future investigation
3. **Use current 51/66 state** for applications needing 100% reliability
4. **Use 58/66 state** for better large-mesh performance (with caveat)

### Long Term
1. **Debug 4-tet allocation systematically:**
   - Add printf-style debugging (counters/flags)
   - Compare tet counts GPU vs CPU
   - Check each step of split kernel
   - Identify exact failure point

2. **Or implement two-phase adjacency update** as simpler alternative

3. **Or accept 58/66 as "good enough"** and document limitations

## Files Modified (Uncommitted)

Lost in git reset, need to recreate:
- `src/coloring.rs` - Graph coloring algorithm
- `src/phase1.rs` - Coloring computation and upload
- `src/gpu/pipelines.rs` - tet_color buffer bindings
- `src/gpu/buffers.rs` - tet_color buffer allocation
- `src/shaders/split.wgsl` - Color filtering
- `src/lib.rs` - Increased buffer limits

## Key Learnings

1. **Graph coloring helps but doesn't solve adjacency** (51→58 improvement, but still broken)
2. **Bidirectional updates need concurrent detection** (empirically proven - corrupts without it)
3. **Concurrent detection needs 4-tet allocation** (formula requires specific memory layout)
4. **4-tet allocation has unknown bug** (blocking the entire cascade)
5. **CUDA's approach is sophisticated** (donation system, insVertVec mapping, complex)
