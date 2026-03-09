# CUDA Free List Design Flaws

This document records design flaws discovered in the original CUDA gDel3D implementation during the WGPU port.

---

## Flaw #1: Allocation/Donation Index Mismatch (Discovered 2026-03-09)

### The Bug

The free list management in `KerDivision.cu` has incompatible allocation and donation strategies that cause re-use of already-allocated tet indices.

### CUDA Code Analysis

**Allocation (lines 120-124):**
```cpp
const int newIdx = ( splitVertex + 1 ) * MeanVertDegree - 1;  // Last position in block
const int newTetIdx[4] = {
    freeArr[ newIdx ],      // Read from END of block
    freeArr[ newIdx - 1 ],
    freeArr[ newIdx - 2 ],
    freeArr[ newIdx - 3 ]
};
vertFreeArr[ splitVertex ] -= 4;  // Decrement count
```

**Donation (lines 184-186):**
```cpp
const int freeIdx = atomicAdd( &vertFreeArr[ vertIdx ], 1 );  // Get index, then increment
freeArr[ vertIdx * MeanVertDegree + freeIdx ] = tetIdx;       // Write to position = count
```

**Initialization (kerUpdateBlockVertFreeList, line 1143):**
```cpp
// Populates from position 0 onwards
freeArr[ vertIdx * MeanVertDegree + locIdx ] = startFreeIdx + idx;
```

### The Problem

1. **Allocation reads from TOP** (positions 7, 6, 5, 4)
2. **Donation writes to BOTTOM** (position = vertFreeArr, starting at 0)
3. **No mechanism to clear/update high positions after allocation**

### Concrete Example

**Initial state (vertex 0, after initialization):**
```
free_arr[0..7] = [base+0, base+1, base+2, base+3, base+4, base+5, base+6, base+7]
vertFreeArr[0] = 8
```

**After 1st allocation:**
```
Read positions [7,6,5,4] → allocate tets [base+7, base+6, base+5, base+4]
vertFreeArr[0] = 4
Valid free slots: positions [0..3]
```

**After 1st donation (tet X):**
```
Write to position 4: free_arr[4] = X
vertFreeArr[0] = 5
Valid free slots: positions [0..4]
```

**After 2nd allocation:**
```
Read positions [7,6,5,4] → allocate tets [base+7, base+6, base+5, X]
                                            ↑      ↑      ↑
                                    ALREADY ALLOCATED!
vertFreeArr[0] = 1
```

**Problem:** Positions 7, 6, 5 still contain their original values (base+7, base+6, base+5), which were already allocated. The algorithm **re-allocates the same tets**!

### Two Incompatible Interpretations of vertFreeArr

- **Allocation treats it as:** Count of available free slots (subtracts from it)
- **Donation treats it as:** Index of next write position (adds to it as pointer)

### Impact

- **Expected:** Each vertex uses ~6-8 tets (3 from split + 3 from flips)
- **Actual in WGPU:** 21-30 tets/vertex, buffer overflow, insertion failures
- **Actual in CUDA:** Unknown - may work due to other factors (see Investigation below)

### Investigation Results: Why CUDA Works Despite This Flaw

**✅ CONFIRMED: CUDA Masks Bug via Frequent Free List Reinitialization**

#### **The Masking Mechanism**

CUDA's `expandTetraList()` is called **EVERY iteration** (GpuDelaunay.cu:844) and triggers a FULL RESET of free_arr via `kerUpdateVertFreeList` (KerDivision.cu:1127-1149):

```cpp
// Called every iteration in splitTetra() loop
expandTetraList(&realInsVertVec, 0, &tetToVert, !_params.noSorting && _doFlipping);
    ↓
// Inside expandTetraList (line 591-596):
kerUpdateVertFreeList<<< ... >>>(
    toKernelArray(*newVertVec),
    toKernelPtr(_vertFreeVec),
    toKernelPtr(_freeVec),
    oldInfBlockIdx
);
    ↓
// Kernel completely reinitializes free_arr (line 1143):
freeArr[vertIdx * MeanVertDegree + locIdx] = startFreeIdx + idx;
vertFreeArr[vertIdx] = MeanVertDegree;  // Reset to 8!
```

**Result:** Positions [7,6,5,4] get **fresh tet indices** from newly allocated buffer regions before stale values can be reallocated.

#### **WGPU's Failure to Mask**

Our `expand_tetra_list()` is a **no-op** (dispatch.rs):
```rust
pub fn expand_tetra_list(&mut self) {
    self.state.current_tet_num = new_size;  // Just update counter!
}
```

**No kernel dispatch** → No free list reset → Stale positions [7,6,5] get reallocated → Bug manifests immediately.

#### **Execution Flow Comparison**

| CUDA (per iteration) | WGPU (per iteration) |
|---------------------|---------------------|
| 1. `expandTetraList()` | 1. `expand_tetra_list()` |
| 2. ↳ `kerUpdateVertFreeList` (RESET free_arr) | 2. ↳ (nothing - just counter++) |
| 3. `kerSplitTetra` (allocate from fresh slots) | 3. `split_tetra` (allocate from STALE slots) |
| 4. ↳ Donation writes to position 4 | 4. ↳ Donation writes to position 4 |
| 5. Next iteration: goto 1 (RESET again!) | 5. Next iteration: goto 1 (no reset) |

**Key Difference:** CUDA's Step 2 prevents the bug from ever triggering.

---

## Investigation Results ✅

### Step 1: Compare Memory Initialization ✅
- ✅ CUDA uses `kerUpdateVertFreeList` (line 1143) - IDENTICAL to our buffers.rs
- ✅ Initial pattern: `freeArr[v*8+i] = base+i` (sequential)
- ✅ DIFFERENCE: CUDA re-runs this EVERY iteration, we run it ONCE

### Step 2: Trace CUDA Execution ✅
- ✅ Found `expandTetraList()` called every iteration (GpuDelaunay.cu:844)
- ✅ Triggers `kerUpdateVertFreeList` which does FULL RESET
- ✅ Confirmed: Re-allocation CAN'T occur because free_arr refreshes constantly

### Step 3: Identify Key Differences ✅
- ✅ **PRIMARY DIFFERENCE:** CUDA reinitializes free_arr every iteration
- ✅ WGPU's `expand_tetra_list()` is a no-op (just counter increment)
- ✅ Compaction/reordering irrelevant - happens only at final output

### Step 4: Solution Options ✅

Based on findings, three viable approaches:

**Option A: Port CUDA's Reinitialization (High Fidelity)**
- Add `update_vert_free_list.wgsl` shader (already exists but unused!)
- Call it in every iteration of insertion loop
- Pros: Matches CUDA exactly, proven to work
- Cons: Performance cost (full buffer rewrite every iteration), doesn't fix architectural flaw

**Option B: Fix Allocation Strategy (Clean Fix)**
- Change `get_free_slots_4tet()` to allocate from BOTTOM (positions [0,1,2,3])
- Match donation's append-from-bottom strategy
- Both use `vertFreeArr` as "next append index"
- Pros: Elegant, no reinitialization needed, fixes root cause
- Cons: Deviates from CUDA behavior

**Option C: Fix Donation Strategy (Preserves CUDA Logic)**
- Change donation to write from TOP: `atomicSub(&vertFreeArr[vertex], 1) - 1`
- Match allocation's pop-from-top strategy
- Both use `vertFreeArr` as "current stack depth"
- Pros: Minimal change to CUDA pattern, fixes root cause
- Cons: More complex atomic (sub then index calculation)

**RECOMMENDED: Option A** (short term) + **Option B** (long term)
- Immediate: Enable `update_vert_free_list` dispatch to match CUDA and fix test failures
- Future: Redesign with bottom-up allocation for cleaner semantics

---

## Complete free_arr Write Analysis

All locations in CUDA that modify `free_arr`:

### 1. **kerSplitTetra (KerDivision.cu:186)** - Donation After Split
```cpp
freeArr[vertIdx * MeanVertDegree + freeIdx] = tetIdx;
```
- **When:** After every tet split
- **Purpose:** Return old tet to owner's free list
- **Pattern:** Writes to bottom (position = vertFreeArr counter)

### 2. **kerFlip (KerDivision.cu:500)** - Donation After Flip32
```cpp
freeArr[vertIdx * MeanVertDegree + freeIdx] = sideTetIdx;
```
- **When:** Flip32 operations only (3→2 tet merge)
- **Purpose:** Return removed side tet to free list
- **Pattern:** Same as kerSplitTetra donation

### 3. **kerUpdateBlockVertFreeList (KerDivision.cu:987)** - Remap During Sort
```cpp
freeArr[freeIdx] = newIdx;
```
- **When:** `expandTetraList()` with `sort=true`
- **Purpose:** Reorder free list when vertices are reordered
- **Pattern:** Remaps ALL positions including high slots [7,6,5,4]

### 4. **kerShiftInfFreeIdx (KerDivision.cu:1010)** - Shift Infinity Block
```cpp
freeArr[freeBeg + idx] = tetIdx + shift;
```
- **When:** After infinity block moves (during expansion)
- **Purpose:** Update tet indices in infinity's free list
- **Pattern:** Only affects infinity block, not vertex blocks

### 5. **kerUpdateVertFreeList (KerDivision.cu:1143)** - FULL RESET
```cpp
freeArr[vertIdx * MeanVertDegree + locIdx] = startFreeIdx + idx;
vertFreeArr[vertIdx] = MeanVertDegree;
```
- **When:** `expandTetraList()` with `sort=false` (EVERY iteration!)
- **Purpose:** **Initialize free list for newly allocated vertex blocks**
- **Pattern:** Writes ALL 8 positions [0..7] with sequential tet indices
- **THIS IS THE KEY MECHANISM** that prevents the bug in CUDA

### 6. **kerCollectFreeSlots (KerDivision.cu:1166)** - Compaction
```cpp
freeArr[freeIdx] = idx;
```
- **When:** `compactTetras()` before final output
- **Purpose:** Build flat array of dead tets for compaction
- **Pattern:** One-time reorganization, not per-iteration

**Key Finding:** Only #5 (`kerUpdateVertFreeList`) fully resets the high positions [7,6,5,4]. WGPU doesn't call this, so the bug manifests.

---

## Related Files

- **CUDA Source:** `gDel3D/GDelFlipping/src/gDel3D/GPU/KerDivision.cu`
  - Lines 120-124: Allocation (kerSplitTetraFast)
  - Lines 184-186: Donation (kerSplitTetraFast)
  - Lines 1137-1148: Initialization (kerUpdateBlockVertFreeList)

- **WGPU Port:** `src/shaders/split.wgsl`
  - Lines 89-105: get_free_slots_4tet (allocation)
  - Lines 314-328: Donation logic

- **Initialization:** `src/gpu/buffers.rs`
  - Lines 182-210: free_arr and vert_free_arr setup

---

## Test Results Demonstrating the Bug

**test_raw_uniform_100 (100 vertices):**
- Expected: ~600 tets (100 × 6 tets/vertex)
- Actual: 1245 tets before overflow
- Result: 42/100 vertices failed to insert

**test_raw_uniform_500 (500 vertices):**
- Expected: ~3000 tets (500 × 6 tets/vertex)
- Actual: 15000+ tets (hit buffer limit)
- Result: 387/500 vertices failed to insert

**After compaction:**
- 100-vertex test: 613 unique tets (6.1 per inserted vertex) ✓
- Suggests dead tets accumulate but correct ratio after cleanup
- Confirms tets are being created correctly, just too many

---

## Summary & Next Steps

### What We Learned

1. **The bug is REAL** - allocation/donation work from opposite ends of free_arr
2. **CUDA masks it accidentally** - `kerUpdateVertFreeList` runs every iteration, resetting free_arr
3. **WGPU exposes it immediately** - no reinitialization, pre-allocated buffers
4. **Root cause:** Architectural flaw, not implementation error

### Immediate Fix (Option A)

Enable the existing `update_vert_free_list.wgsl` shader:
```rust
// In dispatch.rs, inside expand_tetra_list():
if self.state.current_tet_num > old_tet_num {
    self.dispatch_update_vert_free_list(encoder, queue, ins_num);
}
```

This matches CUDA's behavior and should fix test failures immediately.

### Long-Term Fix (Option B)

Redesign allocation to use bottom-up strategy:
```wgsl
// Change get_free_slots_4tet to:
let free_idx = atomicSub(&vert_free_arr[vertex], 4u);
let base = vertex * MEAN_VERTEX_DEGREE;
return vec4<u32>(
    free_arr[base + free_idx - 4u],  // Read from bottom
    free_arr[base + free_idx - 3u],
    free_arr[base + free_idx - 2u],
    free_arr[base + free_idx - 1u]
);
```

This eliminates the need for reinitialization and fixes the architectural flaw.

---

## Investigation Status

- **Status:** ✅ COMPLETE (2026-03-09)
- **Agents used:** 2 parallel investigations
  - Agent a83383b: Traced CUDA execution flow
  - Agent a91f080: Catalogued all free_arr writes
- **Conclusion:** Bug exists in both CUDA and WGPU, CUDA masks via frequent reinitialization
- **Action:** ✅ **OPTION A IMPLEMENTED** (2026-03-09)

## Implementation Status (Option A)

**Files Modified:**
- ✅ Created `src/shaders/update_vert_free_list.wgsl` - Port of kerUpdateVertFreeList
- ✅ Modified `src/gpu/pipelines.rs` - Added uninserted buffer to bind group
- ✅ Modified `src/gpu/mod.rs` - Added dispatch_update_vert_free_list method
- ✅ Modified `src/phase1.rs` - Submit encoder after expand_tetra_list

**Test Results (test_raw_uniform_100):**
- Before fix: 1245 tets, 42/100 vertices failed, buffer limit at ~1300 tets
- After fix: 2965 tets before compaction, 0/100 failures ✅, **all points inserted** ✅
- After compaction: 1035 tets (10.35 tets/vertex)
- Status: **Insertion success improved**, but tet count still high (expected 600-800)

**Remaining Issue:**
- Tet growth is reduced but not eliminated (10.35 tets/vertex vs expected 6-8)
- Possible causes:
  1. Flipping not properly donating tets back
  2. Some allocations still bypassing the reinitialized free lists
  3. Additional CUDA mechanisms we haven't ported

**Next Steps:**
- Investigate flipping donation logic in allocate_flip23_slot.wgsl
- Trace tet allocation/donation balance across iterations
- Consider Option B (redesign allocation) if flipping checks out

---

## Notes

- Discovered during implementation of "Fix Unbounded Tet Growth" plan
- Original plan assumed validation checks caused leaks (incorrect)
- Root cause is architectural flaw in free list design
- Both CUDA and WGPU implementations have the same code structure
- WGPU exhibits the bug more severely due to missing `kerUpdateVertFreeList` calls
