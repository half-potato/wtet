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

### Extended Debugging Session (2026-02-26)

**BREAKTHROUGH: Isolated test works, real pipeline crashes**

Created `tests/test_4tet_step_by_step.rs` with incremental tests:
- ✅ Step 1 (allocate + write): PASSES
- ✅ Step 2 (with old_tet variable + marking dead): PASSES

This proves the allocation logic itself is CORRECT. The bug is in the pipeline integration.

**Debug Methodology:**
1. Added debug counters to track execution flow
2. Instrumented `get_free_slots_4tet` with entry/exit counters
3. Added bounds checking and validation at each step
4. Narrowed crash location with binary search

**Key Findings:**

**Finding #1: Voting System Broken**
```
DEBUG Counters:
  scratch[2] (super-tet splits): 4
```
All 4 insertions target tet 0 (super-tet) in the SAME iteration!
- Expected: Only 1 winner per tet (voting should select 1 point)
- Actual: All 4 points selected for insertion
- Result: 4 threads simultaneously split tet 0 → race condition

**Finding #2: Race Condition on Neighbor Updates**
When 4 threads split the same tet:
1. All read `tets[0]` - OK (read-only)
2. All write `tet_info[0] = 0` - OK (same value)
3. **All update tet 0's neighbors' back-pointers** - RACE!
   - Thread 1: `set_opp_at(neighbor, face, encode_opp(t0_thread1, k))`
   - Thread 2: `set_opp_at(neighbor, face, encode_opp(t0_thread2, k))`
   - Thread 3: `set_opp_at(neighbor, face, encode_opp(t0_thread3, k))`
   - Thread 4: `set_opp_at(neighbor, face, encode_opp(t0_thread4, k))`
   - Multiple DIFFERENT values written to same location!

**Finding #3: GPU Crash on Buffer Write**
Even with neighbor updates disabled and dummy values:
```wgsl
// Dummy values (like isolated test)
let v0 = 0u; let v1 = 1u; let v2 = 2u; let v3 = 3u;

// Write 4 tets
tets[t0] = vec4<u32>(p, v1, v2, v3);
tets[t1] = vec4<u32>(v0, p, v2, v3);
tets[t2] = vec4<u32>(v0, v1, p, v3);
tets[t3] = vec4<u32>(v0, v1, v2, p);

// Mark old tet dead
tet_info[old_tet] = 0u;

// This debug counter NEVER executes
atomicAdd(&counters[2], 1u);
```

Crash happens during or before the tet writes, even though:
- ✅ t0/t1/t2/t3 are validated < 512
- ✅ Buffer size is correct (512 * 16 bytes)
- ✅ Exact same code works in isolated test
- ✅ Using dummy values (no reads from tets[old_tet])
- ✅ No adjacency updates (disabled)

**Finding #4: Allocation Function Works**
```
DEBUG Counters:
  scratch[0] (fn entry): 1
  scratch[1] (count sum): 64  ← read count=64 from vert_free_arr
  scratch[2] (early returns): 0  ← didn't early return
  scratch[3] (array access): 1  ← accessed free_arr
```
`get_free_slots_4tet` executes successfully and returns valid indices.

**Finding #5: Mysterious Difference from Isolated Test**
Isolated test (WORKS):
- Uses `old_tet = 10` (arbitrary tet, not super-tet)
- Manually initialized buffers
- Single dispatch
- Embedded shader in test file

Real pipeline (CRASHES):
- Uses `old_tet = 0` (super-tet)
- Buffers from `GpuBuffers::new()`
- Runs after init, locate, vote, pick_winner kernels
- Shader from `src/shaders/split.wgsl`

**Attempted Fixes (All Failed):**
1. ❌ Bounds check on old_tet (skipping tet 0)
   - Result: Test passes but wrong behavior (skips valid splits)
2. ❌ Validate allocated indices before use
   - Result: Indices are valid, still crashes
3. ❌ Use dummy vertex values instead of reading tets[old_tet]
   - Result: Still crashes
4. ❌ Skip adjacency reads (use INVALID for all)
   - Result: Still crashes
5. ❌ Disable neighbor updates entirely
   - Result: Still crashes
6. ❌ Swap order of operations (write tets before marking dead)
   - Result: Still crashes

**Current Hypothesis:**
Unknown GPU driver issue, buffer synchronization problem, or WGPU validation bug that only manifests when:
- Writing to tets buffer after previous kernels have run
- OR specific pattern of buffer access triggers GPU fault
- OR race condition that's non-deterministic

**Blocking Issue:**
Cannot proceed with 4-tet allocation until root cause identified. This blocks:
- Concurrent split detection
- Bidirectional adjacency updates
- Fixing broken back-pointers
- Achieving 66/66 tests passing

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

## Root Cause Analysis: 4-Tet Allocation Failure

### The Core Problem: Voting System

The **immediate cause** of the crash is that the voting system is broken:
- `pick_winner.wgsl` selects ALL 4 points for insertion in iteration 1
- All 4 target tet 0 (the super-tetrahedron)
- Multiple threads simultaneously write to the same memory locations
- GPU crashes or produces corrupted results

### Why Voting Fails

**Expected behavior:**
1. All 4 points locate to tet 0 (correct - all inside super-tet)
2. Vote kernel: All 4 points vote for tet 0
3. Pick winner: `atomicMin` selects lowest vote value (1 winner)
4. Only 1 insertion in iteration 1
5. Remaining 3 points retry in iteration 2 with new tets

**Actual behavior:**
- 4 insertions in iteration 1 (all targeting tet 0)
- Suggests pick_winner runs 4 times instead of 1?
- OR vote buffer is corrupted?
- OR insert_list is populated incorrectly?

**Diagnostic needed:**
```wgsl
// In pick_winner.wgsl, add:
if tet_idx == 0u {
    atomicAdd(&counters[DEBUG], 1u);  // Count how many times tet 0 wins
}
```
Should be 1, but likely showing 4.

### Secondary Problem: GPU Crash Mystery

Even with only 1 split running (voting partially fixed), the shader crashes:
- Crash happens during basic buffer writes
- Same code works in isolated test
- Suggests deeper issue with buffer state or GPU driver

**Possible causes:**
1. **Buffer aliasing:** Two bind group entries pointing to same buffer?
2. **Validation layer bug:** WGPU or GPU driver validation error
3. **Synchronization:** WAW hazard not properly handled
4. **Out-of-bounds:** Actual crash is elsewhere, debug counters misleading
5. **Race condition:** Non-deterministic GPU behavior

### Next Steps (Prioritized)

**Priority 1: Fix Voting System (Unblocks Progress)**
1. Add debug logging to pick_winner:
   - Count how many tets have votes
   - Log which tets are calling pick_winner
   - Check if tet 0 is selected multiple times
2. Check vote.wgsl for bugs:
   - Is atomicMin working correctly?
   - Are votes being properly packed/unpacked?
3. Verify reset_votes clears tet_vote[0] between iterations
4. If voting is fundamentally broken, consider alternative:
   - Sequential insertion (1 point per iteration)
   - CPU-side conflict resolution

**Priority 2: Alternative to 4-Tet Allocation**
Since 4-tet crashes mysteriously, consider alternatives:
1. **Keep 3-tet allocation, fix concurrent detection differently**
   - Use different mapping formula
   - Store new tet indices in separate buffer
2. **Two-phase split:**
   - Phase 1: Mark tets to split (no allocation)
   - Phase 2: Allocate and split (guaranteed no conflicts)
3. **Accept broken back-pointers for now**
   - Use CPU fixup in phase2
   - Document as known issue
   - Focus on getting more tests passing

**Priority 3: Deep GPU Investigation (Last Resort)**
Only if above approaches fail:
1. Enable WGPU validation layers
2. Test on different GPU/driver
3. Simplify shader to absolute minimum
4. Binary search: which exact instruction crashes?
5. File bug report with WGPU team

## Recommended Path Forward

**Short term (get tests passing):**
1. Fix voting to select only 1 winner per tet
2. Keep 3-tet allocation (works reliably)
3. Implement two-phase adjacency updates
4. Accept some back-pointer issues, fix in phase2

**Long term (proper solution):**
1. Debug and fix 4-tet allocation crash
2. Implement concurrent split detection
3. Full bidirectional adjacency updates
4. Achieve 66/66 tests passing

## Files Modified During Debug Session

**Test files created:**
- `tests/test_4tet_allocation.rs` - Minimal allocation test (PASSES)
- `tests/test_4tet_step_by_step.rs` - Incremental debugging tests (PASS)
- `/tmp/test_debug.rs` - Temporary test file

**Shaders modified:**
- `src/shaders/split.wgsl` - Added debug counters, bounds checks
- Note: Currently in broken state with dummy values and disabled features

**Buffers modified:**
- `src/gpu/buffers.rs` - Fixed vert_free_arr initialization bug

**Tests modified:**
- `src/tests.rs` - Added debug counter output

**Status:** 48/66 tests passing (regression from 51/66 due to debug changes)
