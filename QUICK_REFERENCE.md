# Quick Reference: Bug Status

**TL;DR:** 51/66 tests pass. Blocked by unknown bug in 4-tet allocation.

## The Problem

Large meshes (100+ points) have broken adjacency because we only update one direction:
```wgsl
set_opp_at(my_tet, k, encode_opp(nei_tet, nei_face));  // ✅ We update our pointer
// Missing: set_opp_at(nei_tet, nei_face, ...);        // ❌ Neighbor's back-pointer broken
```

Result: 10-15 test failures with Euler violations (+1 to +8).

## Why We Can't Just Fix It

**The Cascade:**
```
Fix broken back-pointers
  → Need bidirectional updates
    → Need concurrent split detection (avoid race)
      → Need 4-tet allocation (formula requires it)
        → 4-tet allocation has unknown bug ← STUCK HERE
```

## What We Tried

| Approach | Result | Why It Failed |
|----------|--------|---------------|
| Graph coloring + bidirectional | 52/66 | Shared neighbor race |
| One-pass + concurrent detect | 54/66 | Needs 4-tet allocation |
| 4-tet allocation | FAIL | "3/4 points not inserted" |

## The Blocker: 4-Tet Allocation

**What we know:**
- ✅ 3-tet works fine (control test passed)
- ✅ Tet indices from `free_arr` are valid
- ✅ Bounds checks don't trigger
- ✅ Fixed `vert_free_arr` initialization bug
- ❌ Unknown issue prevents insertions

**Most likely culprit:**
Using `old_tet` variable instead of `t0` breaks assumptions elsewhere in code.

## Files to Check

**Core implementation:**
- `src/shaders/split.wgsl` - Split kernel (where bug likely is)
- `src/gpu/buffers.rs` - Buffer initialization
- `src/phase1.rs` - Main insertion loop

**Documentation:**
- `STATUS.md` - Comprehensive status (read this!)
- `CUDA_DIFFERENCES.md` - How we differ from CUDA
- `DEBUGGING_PLAN.md` - Detailed cascade analysis
- `TODO.md` - Path to fixing everything

**CUDA reference:**
- `gDel3D/GDelFlipping/src/gDel3D/GPU/KerDivision.cu` - Original implementation

## Quick Wins

1. **Recover lost work:** Graph coloring (51→58 tests)
   - See TODO.md "Recover Lost Work" section
   - About 1-2 hours of work

2. **Try two-phase adjacency:**
   - Simpler than fixing 4-tet
   - Should get to 66/66
   - See TODO.md "Alternative Path"

## If You Want to Debug 4-Tet

**First test:**
```wgsl
// Keep t0 name, just allocate from free_arr
let old_tet_idx = insert.x;
let slots = get_free_slots_4tet(p);
let t0 = slots.x;  // Try using t0 name
let t1 = slots.y;
// ...
```

Check if variable naming is the issue.

**Instrumentat
ion pattern:**
```wgsl
atomicAdd(&counters[COUNTER_DEBUG_1], 1u);  // Count splits attempted
// ... allocation code ...
atomicAdd(&counters[COUNTER_DEBUG_2], 1u);  // Count splits completed
```

Read back counters to see where splits fail.

## Current Test Status

**Passing (51/66):**
- All small tests (<50 points)
- Most medium tests (50-100 points)
- All special cases (coplanar, cospherical, etc.)

**Failing (15/66):**
- `test_raw_uniform_100` through `test_raw_uniform_1000`
- `test_raw_clustered_100`
- `test_raw_sphere_shell_200`
- Others with 100+ points

## Key Insight

This is a **dependency cascade**, not separate bugs:
- Can't fix adjacency without bidirectional updates
- Can't do bidirectional without concurrent detection
- Can't do concurrent detection without 4-tet allocation
- **4-tet allocation is the root blocker**

Fix that one thing → everything else falls into place.
