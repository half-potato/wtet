# Stub Dispatch Method Analysis

**Generated:** 2026-03-02
**Source:** Parallel agent investigation of CUDA kernels

## Summary

Of the 11 stub dispatch methods, only **4 need implementation** for core functionality. The remaining 7 are for CUDA's dynamic array reordering, which is not needed in WGPU's pre-allocated buffer design.

---

## ✅ IMPLEMENT THESE (4 methods)

### 1. dispatch_collect_free_slots
- **CUDA:** KerDivision.cu:1152-1168
- **Shader:** ✅ collect_free_slots.wgsl (correct)
- **Purpose:** Collects dead tet indices for compaction
- **Used in:** compactTetras() - final mesh cleanup
- **Status:** CRITICAL - needed for final output

### 2. dispatch_make_compact_map
- **CUDA:** KerDivision.cu:1171-1188
- **Shader:** ✅ make_compact_map.wgsl (correct)
- **Purpose:** Creates old→new index mapping for compaction
- **Used in:** compactTetras() - after collect_free_slots
- **Status:** CRITICAL - needed for final output

### 3. dispatch_compact_tets
- **CUDA:** KerDivision.cu:1191-1233
- **Shader:** ✅ compact_tets.wgsl (correct, uses 5-bit TetOpp)
- **Purpose:** Physically moves alive tets to compact array
- **Used in:** compactTetras() - final step
- **Status:** CRITICAL - needed for final output

### 4. dispatch_mark_special_tets
- **CUDA:** KerDivision.cu:834-869
- **Shader:** ✅ mark_special_tets.wgsl (correct)
- **Purpose:** Clears OPP_SPECIAL flags between fast/exact flipping
- **Used in:** Between fast and exact flipping phases
- **Status:** CRITICAL - needed for correct flipping

---

## ❌ NOT NEEDED (7 methods - CUDA reordering only)

These are for CUDA's sorting/reordering path when vertices are compacted. WGPU uses pre-allocated buffers and doesn't need vertex reordering.

### 5. dispatch_update_block_vert_free_list
- **Reason:** Free lists pre-initialized in buffers.rs
- **CUDA Path:** Sorting path only (expandTetraList line 541)
- **WGPU:** Pre-allocation makes this unnecessary

### 6. dispatch_update_block_opp_tet_idx
- **Reason:** No vertex reordering in WGPU
- **CUDA Path:** Sorting path only (expandTetraList line 504)
- **WGPU:** Minimal design doesn't use sorting path

### 7. dispatch_shift_inf_free_idx
- **Reason:** Infinity block doesn't move in pre-allocated design
- **CUDA Path:** Both paths (lines 550, 574)
- **WGPU:** Static allocation = no shifting needed

### 8. dispatch_update_tet_idx
- **Reason:** No vertex reordering in WGPU
- **CUDA Path:** Sorting path only (line 511)
- **WGPU:** Minimal design doesn't use sorting path

### 9. dispatch_shift_opp_tet_idx
- **Reason:** No dynamic array growth in WGPU
- **CUDA Path:** Non-sorting path (line 562)
- **WGPU:** Pre-allocated buffers = no shifting

### 10. dispatch_shift_tet_idx
- **Reason:** No dynamic array growth in WGPU
- **CUDA Path:** Non-sorting path (line 568)
- **WGPU:** Pre-allocated buffers = no shifting

### 11. dispatch_make_reverse_map
- **Reason:** No vertex sorting in WGPU
- **CUDA Path:** Sorting path only (line 495)
- **WGPU:** Minimal design doesn't use sorting path

---

## Pipeline Integration

**compactTetras() flow:**
1. Compute prefix sum (CPU/Thrust → needs WGPU equivalent)
2. dispatch_collect_free_slots
3. dispatch_make_compact_map
4. dispatch_compact_tets

**markSpecialTets() flow:**
1. dispatch_mark_special_tets (between fast/exact flipping)

---

## Implementation Notes

All 4 required shaders are **already ported and correct**:
- Use 5-bit TetOpp encoding where needed ✅
- Match CUDA algorithms exactly ✅
- Proper atomic operations for concurrent access ✅

Only need to:
1. Add pipeline fields to Pipelines struct (if not already present)
2. Implement dispatch wrapper functions
3. Wire into main pipeline (compactTetras for final output, markSpecialTets in flipping loop)

---

## Agent Reports

Full detailed reports from 7 parallel agents available in this session's transcript.
