# Debugging Plan: The Cascade Dependency

## Current State: 51/66 Tests Passing

## The Dependency Cascade

We discovered a **cascade of dependencies** blocking the fix:

```
Issue #1: Broken Back-Pointers (10-15 test failures)
    Root Cause: One-way adjacency updates
    Evidence: Euler violations +1 to +8 in large meshes
    ↓ requires
Issue #2: Bidirectional Adjacency Updates
    Attempted: Graph coloring + bidirectional → 52/66 (WORSE)
    Problem: Shared-neighbor race condition
    ↓ requires
Issue #3: Concurrent Split Detection
    Attempted: One-pass + detection → 54/66 (WORSE)
    Problem: Requires 4-tet allocation for formula
    ↓ requires
Issue #4: 4-Tet Allocation
    Attempted: Multiple approaches → MASSIVE FAILURES
    Status: **BLOCKED - Unknown bug**
```

**We are stuck at Issue #4, which blocks everything above it.**

## Evidence: Graph Coloring Doesn't Solve Adjacency

**Test Results:**
- Simple coloring (tet_idx % 4): 58/66 tests
- Proper graph coloring (greedy): 58/66 tests (no improvement!)
- Bidirectional + coloring: 52/66 tests (WORSE)

**Why Coloring Fails:**
- Graph coloring ensures NO TWO ADJACENT tets have same color
- BUT multiple SAME-COLOR tets can share a COMMON NEIGHBOR (different faces)
- Example:
  ```
  Tet A (color 0) ←→ Tet C (color 1) [face 0]
  Tet B (color 0) ←→ Tet C (color 1) [face 2]

  When A and B split:
  - Both try to update C's adjacency
  - Different faces, but same tet
  - Empirically causes corruption
  ```

## Evidence: Concurrent Detection Needs One-Pass

**Test Results:**
- 4-color + concurrent detection: 55/66 tests
- One-pass + concurrent detection: 54/66 tests

**Why 4-Color Fails:**
Sequential color passes make `tet_to_vert` stale:
- Color 0 splits → marks tets in `tet_to_vert`
- Color 1 splits → checks `tet_to_vert[neighbor]`
- If neighbor was color 0: already finished splitting!
- Reads wrong `free_arr` values → corruption

## Evidence: Concurrent Detection Needs 4-Tet

CUDA's formula: `freeArr[block_top - face]`

This assumes:
- All 4 new tets allocated sequentially from `free_arr[block_top]` downwards
- Tet with external face k is at `free_arr[block_top - k]`

Our 3-tet allocation:
```
k=0: t0 (REUSED from insert_list, NOT from free_arr)
k=1: t1 = free_arr[block_top]
k=2: t2 = free_arr[block_top - 1]
k=3: t3 = free_arr[block_top - 2]
```

Formula `free_arr[block_top - 0]` would give t1, not t0!

## Issue #4 Deep Dive: 4-Tet Allocation Bug

**Symptom:** "3 out of 4 points fail to insert" in trivial 4-point test

**Implementation:**
```wgsl
let slots = get_free_slots_4tet(vertex);
let t0 = slots.x;  // All from free_arr
let t1 = slots.y;
let t2 = slots.z;
let t3 = slots.w;
tet_info[old_tet] = 0u;  // Mark original dead
```

**Debugging Done:**
1. ✅ Verified tet indices are valid ([64, 63, 62, 61] for vertex 0)
2. ✅ Verified bounds checking doesn't trigger (all < max_tets)
3. ✅ Verified 3-tet allocation still works (control test)
4. ✅ Found and fixed `vert_free_arr` initialization bug
5. ❌ Insertions still fail

**Suspected Causes:**
1. Using `old_tet` variable instead of `t0` breaks assumptions
2. Other code expects `t0 == insert_list[idx].x`
3. Marking old tet dead causes timing issue
4. Unknown pipeline interaction

## Debugging Strategy

### Phase 1: Verify Broken Back-Pointers

Add diagnostic shader to count broken adjacency:

```wgsl
// count_broken_adjacency.wgsl
@compute fn count_broken_adjacency() {
    if tet is alive {
        for each face {
            nei_tet = my_opp[face]
            if nei_tet is alive {
                back_opp = nei_tet.opp[nei_face]
                if back_opp != (me, my_face) {
                    atomicAdd(&broken_count, 1)
                }
            }
        }
    }
}
```

**Expected:** Large tests have hundreds of broken back-pointers

### Phase 2: Fix Bidirectional Updates Safely

**Option A: Two-Phase Update (Recommended)**
```rust
// After all color splits complete:
1. dispatch_collect_adjacency_updates()  // Each tet writes updates to temp buffer
2. dispatch_apply_adjacency_updates()    // Apply all updates from temp buffer
```

Eliminates races because:
- Phase 1: Only reads, writes to separate buffer
- Phase 2: Only writes to tet_opp, no reads

**Option B: Post-Split Fixup**
```wgsl
// fixup_back_pointers.wgsl
@compute fn fixup_back_pointers() {
    for each alive tet {
        for each face {
            nei_tet = my_opp[face]
            check if nei_tet.opp[nei_face] points back to me
            if not, fix it
        }
    }
}
```

Simpler but may have conflicts if multiple tets try to fix same neighbor.

### Phase 3: Test Fix

1. Implement Option A (two-phase update)
2. Run `test_raw_uniform_100` - should pass if hypothesis correct
3. Run full suite - expect 66/66 if this was the only issue

### Phase 4: If Still Failing

If two-phase update doesn't fix all tests, investigate 3-tet reuse:

1. Compare tet counts: GPU vs expected (6.5x points)
2. Check free_arr state after splits
3. Look for duplicate tet allocations

## Code Locations

**One-way update to fix:**
- `src/shaders/split.wgsl:180-188` - external adjacency update loop

**Add diagnostics:**
- Create `src/shaders/count_broken_adjacency.wgsl`
- Add to pipelines.rs and dispatch.rs

**Two-phase implementation:**
- Create `adjacency_updates` buffer (size: max_tets * 4 * 2 u32s)
- Modify split shader to write to adjacency_updates
- Create apply shader to read and apply updates

## Quick Test

To verify hypothesis quickly, try **disabling one-way updates entirely**:

```wgsl
// In split.wgsl, comment out the adjacency update loop
// for (var k = 0u; k < 4u; k++) { ... }
```

Then use phase2 CPU adjacency rebuild. If tests pass, confirms adjacency is the issue.
