# Critical Implementation Flaws in WGPU Port

This document tracks fundamental deviations from the CUDA implementation that constitute design flaws, not intentional optimizations.

---

## FLAW #1: CPU-Side Compaction Instead of GPU

**Location:** `src/phase1.rs:457-478`

**CUDA Implementation:**
```cpp
// GpuDelaunay.cu:360, 478, 1069
compactBothIfNegative( _vertTetVec, _vertVec );

// ThrustWrapper.cu:244-265 - RUNS ON GPU
void compactBothIfNegative(IntDVec& vec0, IntDVec& vec1)
{
    const IntZipDIter newEnd =
        ::mgx::thrust::remove_if(
            ::mgx::thrust::make_zip_iterator( ::mgx::thrust::make_tuple( vec0.begin(), vec1.begin() ) ),
            ::mgx::thrust::make_zip_iterator( ::mgx::thrust::make_tuple( vec0.end(), vec1.end() ) ),
            IsIntTuple2Negative() );

    vec0.erase( ::mgx::thrust::get<0>( endTuple ), vec0.end() );
    vec1.erase( ::mgx::thrust::get<1>( endTuple ), vec1.end() );
}
```

**WGPU Implementation:**
```rust
// src/phase1.rs:457-478 - RUNS ON CPU
state
    .uninserted
    .retain(|v| !inserted_verts.contains(v));

let mut new_vert_tet = Vec::with_capacity(state.uninserted.len());
for (old_pos, &tet_idx) in old_vert_tet.iter().enumerate() {
    if !inserted_positions.contains(&old_pos) {
        new_vert_tet.push(tet_idx);
    }
}

queue.write_buffer(
    &state.buffers.vert_tet,
    0,
    bytemuck::cast_slice(&new_vert_tet),
);
```

**The Flaw:**
- CUDA: Uses GPU parallel thrust::remove_if with zip_iterator over both vectors simultaneously
- WGPU: Uses CPU sequential Vec::retain and loops
- CUDA: Zero CPU-GPU synchronization cost (stays on GPU)
- WGPU: Forces GPU→CPU readback, CPU processing, CPU→GPU writeback

**Performance Impact:**
- CPU-GPU round trip latency (microseconds to milliseconds)
- Sequential CPU processing instead of parallel GPU processing
- Pipeline bubble: GPU sits idle while CPU processes
- Scales poorly with large point sets (10k+ points)

**Why This Happened:**
Thrust's `remove_if` with zip iterators is non-trivial to port to WGPU compute shaders. The implementer took the "simple" path of doing it on CPU instead.

**What Should Happen:**
Implement GPU-side parallel compaction using prefix sum + scatter pattern (already exists in compact_tets.wgsl for tet compaction).

---

## FLAW #2: Missing TET_EMPTY Bit Implementation

**CUDA tet_info Bit Layout** (CommonTypes.h:384-388):
```
76543210
     ^^^ 0: Dead      1: Alive     (bit 0)
     ||_ 0: Checked   1: Changed   (bit 1)
     |__ 0: NotEmpty  1: Empty     (bit 2)
```

**CUDA Accessors:**
```cpp
bool isTetEmpty(char c) { return isBitSet(c, 2); }
void setTetEmptyState(char& c, bool b) { setBitState(c, 2, b); }
```

**WGPU Implementation:**
```rust
// src/types.rs:103-104 - MISSING BIT 2
pub const TET_ALIVE: u32 = 1 << 0;
pub const TET_CHANGED: u32 = 1 << 1;
// TET_EMPTY: u32 = 1 << 2;  // NOT DEFINED
```

**CUDA Usage of TET_EMPTY:**

### 1. kerMarkTetEmpty (KerDivision.cu:788-795)
Sets ALL tets to empty=true at start of each iteration:
```cpp
__global__ void kerMarkTetEmpty(KerCharArray tetInfoVec)
{
    for (int idx = getCurThreadIdx(); idx < tetInfoVec._num; idx += getThreadNum())
        setTetEmptyState(tetInfoVec._arr[idx], true);
}
```

**Purpose:** Mark all tets as "not yet participating in flips" at iteration start.

### 2. kerSplitPointsFast (KerPredicates.cu:232, 276)
Sets tets to empty=false when they contain uninserted vertices:
```cpp
if (splitVertIdx == INT_MAX) {  // Tet not split
    setTetEmptyState(tetInfoArr[tetIdx], false);
    continue;
}
// ... later during decision tree navigation:
setTetEmptyState(tetInfoArr[face], false);  // Mark visited face
```

**Purpose:** Track which tets contain uninserted vertices (empty=false) vs only super-tet vertices (empty=true).

### 3. kerFlip (KerDivision.cu:514-527)
Checks if flip involves only "clean" tets:
```cpp
bool tetEmpty = isTetEmpty(tetInfoArr[botTetIdx]) &&
                isTetEmpty(tetInfoArr[topTetIdx]);

if (fType == Flip32 && tetEmpty)
    tetEmpty = isTetEmpty(tetInfoArr[sideTetIdx]);

// Record the flip
FlipItem flipItem = { botTet._v[0], botTet._v[1], botTet._v[2],
    botTet._v[3], topTet._v[topTetVi],
    botTetIdx, topTetIdx, sideTetIdx };

if (tetEmpty)
    flipItem._v[0] = -1;  // Mark as special

// New tets inherit empty state
setTetEmptyState(botTetState, tetEmpty);
setTetEmptyState(topTetState, tetEmpty);
if (fType == Flip32)
    setTetEmptyState(sideTetState, tetEmpty);
```

**Purpose:**
- Distinguish "clean" flips (only super-tet boundary) from "dirty" flips (involve uninserted vertices)
- FlipItem with v[0] = -1 marks special/empty flips for different processing
- Newly created tets inherit the empty state to propagate the property

**WGPU Status:**
- ❌ TET_EMPTY constant not defined anywhere
- ❌ No equivalent to kerMarkTetEmpty (never sets tets to empty=true)
- ❌ split_points.wgsl does NOT set empty=false like KerSplitPointsFast
- ❌ flip.wgsl does NOT check tetEmpty or set FlipItem._v[0] = -1
- ❌ flip.wgsl does NOT inherit/propagate empty state to new tets

**Impact:**

1. **Algorithm Correctness:** Cannot distinguish flips involving uninserted vertices from clean boundary flips. This may affect:
   - Convergence behavior (which flips to prioritize)
   - Phase transition logic (when to switch from GPU to CPU star splaying)
   - Flip validation (some flips may need different handling based on empty state)

2. **Debugging:** Missing semantic information makes it harder to trace why certain flips succeed/fail

3. **Potential Bugs:** If any downstream logic in CUDA depends on FlipItem._v[0] == -1 marker or empty state checks, WGPU will have wrong behavior

**Root Cause Investigation:**

The TET_EMPTY bit was never ported during initial WGPU implementation. Investigation reveals:

1. **CUDA Call Sites:**
   - `kerMarkTetEmpty` (GpuDelaunay.cu:1013) - Called at start of EACH insertion iteration
   - Sets all tets to empty=true (meaning "not containing uninserted vertices")

2. **Where It's Set to False:**
   - `kerSplitPointsFast` (KerPredicates.cu:232, 276) - Marks tets containing uninserted vertices
   - Sets empty=false when tet contains or is visited during decision tree

3. **Where It's Consumed:**
   - `kerFlip` (KerDivision.cu:514-527) - Checks if all tets in flip are empty
   - If all empty → sets `flipItem._v[0] = -1` (special marker)
   - New tets inherit empty state from originals

4. **Downstream Usage:**
   - `kerUpdateFlipTrace` (KerDivision.cu:755) - Skips flips with `_v[0] == -1`
   - Purpose: Don't trace "clean" boundary flips (no uninserted vertices involved)
   - **CRITICAL FINDING:** `update_flip_trace.wgsl:85` ALREADY has this check!
   ```wgsl
   if flip_item.v[0] == -1 {
       return;  // Skip empty flips
   }
   ```
   - BUT `flip.wgsl` never sets `v[0] = -1` because TET_EMPTY isn't tracked

**Actual Impact:**
- **Correctness:** Minimal - flip tracing still works, just does extra work
- **Performance:** Minor - traces ALL flips instead of skipping "empty" ones
- **Semantics:** Moderate - loses information about which flips involve uninserted vertices
- **Bug Potential:** `update_flip_trace.wgsl` expects the marker but never gets it

**Why This Happened:**
The implementer ported `update_flip_trace.wgsl` faithfully (including the v[0] == -1 check), but didn't realize that `flip.wgsl` needs to SET that marker based on TET_EMPTY state. Only TET_ALIVE and TET_CHANGED were ported because they're more obviously critical.

**What Should Happen:**
1. Define `TET_EMPTY: u32 = 1 << 2` in types.rs
2. Create mark_tet_empty.wgsl kernel (sets all tets to empty=true)
3. Call mark_tet_empty at start of each insertion iteration (before split_points)
4. Update split_points.wgsl to set empty=false when visiting tets (lines 232, 276 equivalent)
5. Update flip.wgsl to:
   - Check tetEmpty for all tets in flip (read tet_info & TET_EMPTY)
   - Set `flipItem._v[0] = -1` when tetEmpty is true
   - Inherit empty state in new tets (set/clear TET_EMPTY bit)

---

## Investigation Priority

**FLAW #1 (CPU Compaction):** Medium priority
- Performance issue, not correctness issue
- Only affects large point sets (1k+ points)
- Can be fixed by porting compact logic from compact_tets.wgsl
- **Estimated impact:** 2-5ms per iteration on large datasets, GPU sits idle during CPU processing

**FLAW #2 (TET_EMPTY):** Low-Medium priority (after investigation)
- **NOT a correctness issue** - flip tracing still works without it
- **Performance issue:** Extra flip tracing work (traces all flips instead of just "dirty" ones)
- **Semantic drift:** Loses information about which tets contain uninserted vertices
- **Code inconsistency:** `update_flip_trace.wgsl` expects the marker but never gets it
- Required for 100% faithful CUDA port
- **Estimated impact:** Likely negligible on small-medium datasets, may matter on 10k+ points with many "clean" boundary flips

---

## Root Cause Analysis

Both flaws stem from the same pattern: **Taking shortcuts during porting instead of faithful 1:1 translation.**

**FLAW #1:** "It's simpler to do on CPU" → Wrong. CUDA does it on GPU for a reason.

**FLAW #2:** "Only TET_ALIVE matters" → Wrong. CUDA uses 3 bits for a reason.

**Lesson:** When porting performance-critical GPU algorithms, **ALWAYS port ALL features first, optimize later.** Shortcuts create semantic drift that's hard to debug.

---

## Summary

Both flaws have been investigated and documented:

### FLAW #1: CPU Compaction
- **What:** thrust::remove_if (GPU) ported as Vec::retain (CPU)
- **Where:** phase1.rs:457-478 (compactBothIfNegative logic)
- **Why:** Porting thrust zip iterators to WGPU was non-trivial
- **Impact:** CPU-GPU synchronization overhead, sequential processing
- **Fix:** Implement GPU-side prefix sum + scatter (like compact_tets.wgsl)

### FLAW #2: Missing TET_EMPTY
- **What:** Bit 2 of tet_info never ported from CUDA
- **Where:** Not defined in types.rs, not set in any shader
- **Why:** Only TET_ALIVE/TET_CHANGED seemed critical, implementer didn't trace the dependency chain to update_flip_trace.wgsl
- **Impact:** `update_flip_trace.wgsl` expects FlipItem._v[0] == -1 marker but never gets it, causing extra flip tracing work
- **Fix:** Implement mark_tet_empty.wgsl, update split_points.wgsl and flip.wgsl to track empty state

Neither flaw is a critical correctness bug, but both represent shortcuts that compromise performance and semantic fidelity to the original CUDA implementation.
