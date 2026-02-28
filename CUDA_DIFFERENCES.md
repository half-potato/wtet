# Differences from CUDA gDel3D Implementation

This document explains all major deviations from the original CUDA gDel3D code and their justifications.

## 1. Flat Atomic tet_opp Array

**CUDA:**
```cuda
TetOpp oppArr[max_tets];  // vec4 structure
oppArr[tetIdx].setOpp(face, neiTet, neiFace);
```

**WGSL:**
```wgsl
@group(0) @binding(1) var<storage, read_write> tet_opp: array<atomic<u32>>;
// Indexed as [tet_idx * 4 + face]
fn set_opp_at(tet_idx: u32, face: u32, val: u32) {
    atomicStore(&tet_opp[tet_idx * 4u + face], val);
}
```

**Justification:**
- WGPU requires explicit `atomic<T>` types for concurrent access
- Cannot have `array<vec4<atomic<u32>>>` - must flatten to 1D array
- This is a necessary adaptation to WGSL's type system

**Status:** ✅ Working correctly, well-tested

---

## 2. 3-Tet Allocation (Reusing t0)

**CUDA (KerDivision.cu:121):**
```cuda
const int newTetIdx[4] = { freeArr[newIdx], freeArr[newIdx-1], 
                           freeArr[newIdx-2], freeArr[newIdx-3] };
// Allocates 4 NEW tets, old tet donated back (lines 182-188)
```

**WGSL:**
```wgsl
fn pop_free_slots_for_split(vertex: u32) -> vec3<u32> {
    let block_top = (vertex + 1u) * MEAN_VERTEX_DEGREE - 1u;
    var slots: vec3<u32>;
    slots.x = free_arr[block_top];
    slots.y = free_arr[block_top - 1u];
    slots.z = free_arr[block_top - 2u];
    atomicSub(&vert_free_arr[vertex], 3u);
    return slots;
}
// Original tet t0 is REUSED, only allocate 3 new tets
```

**Justification:**
- Attempted CUDA's 4-tet approach but encountered issues:
  - Donation overwrites recently-allocated slots
  - Reading stale tet indices from free_arr
  - Persistent segfaults despite extensive bounds checking
- 3-tet reuse avoids donation complexity
- Net effect same: +3 tets per insertion

**Status:** ⚠️ Working (51 tests pass) but not faithful to CUDA
**Impact:** Blocks concurrent split detection (formula requires 4-tet)

**4-Tet Allocation Attempts:**
Multiple attempts to implement CUDA's 4-tet allocation all failed:

Attempt 1: Basic 4-tet (no donation)
```wgsl
let slots = get_free_slots_4tet(vertex);  // Read 4 from free_arr
let t0 = slots.x;
let t1 = slots.y;
let t2 = slots.z;
let t3 = slots.w;
tet_info[old_tet] = 0u;  // Mark original dead
```
Result: **"3 out of 4 points fail to insert"** even in trivial test

Bugs Found:
1. ✅ Fixed: `vert_free_arr` initialized to MEAN_VERTEX_DEGREE for all vertices
   - Last vertices get fewer tets but claimed to have 64
   - Fixed by counting actual tets per vertex during init
2. ❌ Unresolved: Unknown issue prevents insertions
   - Tet indices from free_arr are valid
   - Bounds checking doesn't trigger
   - 3-tet allocation works fine (empirically verified)
   - Suspected: Using `old_tet` instead of `t0` breaks assumptions

**This blocks the entire cascade:**
- 4-tet needed for concurrent split detection formula
- Concurrent detection needed for safe bidirectional updates
- Bidirectional updates needed to fix 10-15 test failures

---

## 3. 4-Color Splitting (Concurrent Split Avoidance)

**CUDA (KerDivision.cu:148-163):**
```cuda
// Check if neighbour has split
const int neiSplitIdx = tetToVert[neiTetIdx];
if (neiSplitIdx == INT_MAX) { // Un-split
    oppArr[neiTetIdx].setOpp(neiTetVi, newTetIdx[vi], 3);
} else { // Split - find new tet
    const int neiSplitVert = vertArr[neiSplitIdx];
    const int neiFreeIdx = (neiSplitVert + 1) * MeanVertDegree - 1;
    neiTetIdx = freeArr[neiFreeIdx - neiTetVi];
    neiTetVi = 3;
}
// Handles concurrent splits on-the-fly
```

**WGSL:**
```wgsl
// Color filtering: only process tets with matching color
let my_color = tet_color[t0];
if my_color != color { return; }

// Split in 4 separate passes (colors 0, 1, 2, 3)
for color in 0..4 {
    dispatch_split(encoder, queue, num_inserted, color);
}
```

**Justification:**
- CUDA's concurrent split detection caused persistent segfaults:
  - Accessing `free_arr[nei_block_top - nei_tet_face]` fails bounds checks
  - Race conditions on adjacency updates
  - Complex pointer arithmetic in WGSL
- 4-color splitting guarantees no two adjacent tets split simultaneously
- Eliminates need for concurrent detection

**Status:** ✅ Working, tested with both simple `tet_idx % 4` and proper graph coloring
**Impact:** Slower (4 passes instead of 1) but deterministic and safe

---

## 4. One-Way Adjacency Updates (Failed to Port Concurrent Split Detection)

**CUDA:**
```cuda
oppArr[neiTetIdx].setOpp(neiTetVi, newTetIdx[vi], 3);  // Update neighbor
newOpp.setOpp(3, neiTetIdx, neiTetVi);                 // Update self
// Bidirectional adjacency updates
```

**WGSL:**
```wgsl
// Set our pointer to neighbor (one-way update)
set_opp_at(new_tets[k], k, encode_opp(nei_tet, nei_face));
// Does NOT update neighbor's back-pointer
```

**How CUDA Handles This:**
```cuda
// Check if neighbor is also splitting
const int neiSplitIdx = tetToVert[neiTetIdx];

if (neiSplitIdx == INT_MAX) {
    // Neighbor NOT splitting → safe to update directly
    oppArr[neiTetIdx].setOpp(neiTetVi, newTetIdx[vi], 3);
} else {
    // Neighbor IS splitting → find which NEW tet to point to
    const int neiSplitVert = vertArr[neiSplitIdx];
    const int neiFreeIdx = (neiSplitVert + 1) * MeanVertDegree - 1;
    neiTetIdx = freeArr[neiFreeIdx - neiTetVi];  // Get correct new tet
    neiTetVi = 3;
}
newOpp.setOpp(3, neiTetIdx, neiTetVi);  // Point to (possibly new) neighbor
```

**CUDA's approach works because:**
1. **One simultaneous pass** - All splits happen at once (not 4 sequential batches)
2. **Concurrent split detection** - Uses `tetToVert` to check if neighbor is splitting
3. **Formula finds correct new tet** - `freeArr[neiFreeIdx - neiTetVi]` maps face to new tet
4. **4-tet allocation** - Formula assumes all 4 new tets are allocated from free_arr

**Why Our Implementation Failed:**
- **Attempt 1: 4-color with bidirectional** (no concurrent detection)
  - Result: 52/66 tests (down from 58)
  - Issue: Multiple same-color tets share neighbors → write conflicts

- **Attempt 2: One-pass with concurrent detection**
  - Result: 54/66 tests (down from 58)
  - Issue: 3-tet allocation breaks formula, bounds checking failures

- **Attempt 3: 4-color with concurrent detection**
  - Result: 55/66 tests (down from 58)
  - Issue: Sequential color passes create stale `tet_to_vert` data
  - When color 1 checks if neighbor is splitting, color 0 already finished

**Root Cause:** We use **3-tet allocation** (reuse t0) and **4-color batching**, but CUDA uses **4-tet allocation** and **one-pass**. The concurrent split detection formula only works with CUDA's approach.

**Status:** ⚠️ Working (58/66) but leaves broken back-pointers
**Impact:** 10 test failures due to missing adjacency edges
**Euler violations:** +1 to +8 (small topology errors in large meshes)

**Possible Solutions:**
1. **Implement CUDA's 4-tet allocation** - Match CUDA exactly, enable concurrent detection
2. **Two-phase adjacency update** - Collect updates in buffer, then apply (see DEBUGGING_PLAN.md)
3. **Accept current state** - 58/66 passing is good enough for most use cases

---

## 5. vert_free_arr as Atomic

**CUDA (KerDivision.cu:124, 187):**
```cuda
vertFreeArr[splitVertex] -= 4;  // Regular subtraction (line 124)
const int freeIdx = atomicAdd(&vertFreeArr[vertIdx], 1);  // Atomic for donation (187)
```

**WGSL:**
```wgsl
@group(0) @binding(6) var<storage, read_write> vert_free_arr: array<atomic<u32>>;
atomicSub(&vert_free_arr[vertex], 3u);  // Must be atomic
```

**Justification:**
- WGSL doesn't allow mixed atomic/non-atomic access to same buffer
- If declared as `array<atomic<u32>>`, ALL accesses must use atomic ops
- CUDA can mix because of relaxed memory model

**Status:** ✅ Correct, no impact

---

## 6. Mark Split Pre-Pass

**CUDA:**
```cuda
// Embedded in split kernel (KerDivision.cu)
// tetToVert populated as needed
```

**WGSL:**
```wgsl
// Separate mark_split shader before split
@compute fn mark_split() {
    let tet_idx = insert_list[idx].x;
    let vert_idx = insert_list[idx].y;
    tet_to_vert[tet_idx] = vert_idx;
}
```

**Justification:**
- Cleaner separation of concerns
- Ensures tet_to_vert is populated before any split reads it
- No functional difference

**Status:** ✅ Correct

---

## Summary of Root Causes for Test Failures

Based on this analysis, the **10 failing tests** are likely due to:

### Primary Suspect: One-Way Adjacency (#4)
- **Broken back-pointers accumulate** in large meshes
- Small Euler violations (V-E+F-T = +1 to +8) consistent with missing edges
- Simple fix: Implement proper bidirectional updates without races

### Secondary Suspect: 3-Tet Reuse (#2)
- Doesn't match CUDA's 4-tet allocation + donation
- May cause subtle issues in free list management
- Harder fix: Port CUDA's donation logic correctly

### Not the Cause:
- ❌ Coloring scheme (tested: no difference between simple/proper)
- ❌ Flat atomic array (working correctly)
- ❌ Mark split pre-pass (no impact)

---

## Recommended Next Steps

1. **Fix bidirectional adjacency** (most likely fix):
   - Implement a separate "adjacency fixup" pass after all colors finish
   - Or use a two-phase update: write to temp buffer, then swap

2. **Debug one failing test** in detail:
   - Use `test_raw_uniform_100` (smallest failure)
   - Compare tet adjacency with CPU rebuild
   - Identify specific broken pointers

3. **Port CUDA's 4-tet donation** (if adjacency fix insufficient):
   - Understand CUDA's `insVertVec` mapping
   - Implement proper block ownership tracking
   - Add refill mechanism if needed
