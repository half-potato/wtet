# Missing Kernels Implementation Progress

## ✅ Completed Shader Ports (14 kernels - Not Yet Integrated)

### Memory Management / Compaction (3 kernels)
1. **collect_free_slots.wgsl** - Port of `kerCollectFreeSlots` (KerDivision.cu:1152-1169)
   - Collects dead tet indices into free array for reuse

2. **make_compact_map.wgsl** - Port of `kerMakeCompactMap` (KerDivision.cu:1171-1189)
   - For alive tets beyond newTetNum, maps old index → new index using free slots

3. **compact_tets.wgsl** - Port of `kerCompactTets` (KerDivision.cu:1191-1234)
   - Moves alive tets to new positions and updates adjacency

### Flip Management (2 kernels)
4. **mark_special_tets.wgsl** - Port of `kerMarkSpecialTets` (KerDivision.cu:834-870)
   - Clears special adjacency markers (bit 3 in TetOpp)
   - Marks tets as Changed (bit 1 in tet_info)

5. **update_flip_trace.wgsl** - Port of `kerUpdateFlipTrace` (KerDivision.cu:741-785)
   - Builds flip history chain for tet relocation
   - Links flips in tetToFlip array

### Block-Based Allocation Management (2 kernels)
6. **update_block_vert_free_list.wgsl** - Port of `kerUpdateBlockVertFreeList` (KerDivision.cu:956-990)
   - Updates free list for vertex blocks after reordering
   - Handles both new and existing vertices

7. **update_block_opp_tet_idx.wgsl** - Port of `kerUpdateBlockOppTetIdx` (KerDivision.cu:1015-1048)
   - Remaps adjacency tet indices after block reordering
   - Uses orderArr mapping for indices < oldInfBlockIdx

### Index Shifting / Remapping (4 kernels)
8. **shift_inf_free_idx.wgsl** - Port of `kerShiftInfFreeIdx` (KerDivision.cu:992-1013)
   - Shifts infinity vertex free list indices by offset

9. **update_tet_idx.wgsl** - Port of `kerUpdateTetIdx` (KerDivision.cu:1050-1076)
   - Remaps tet indices after block reordering
   - Preserves sign of negative indices

10. **shift_opp_tet_idx.wgsl** - Port of `kerShiftOppTetIdx` (KerDivision.cu:1078-1103)
    - Shifts adjacency tet indices >= start by shift amount

11. **shift_tet_idx.wgsl** - Port of `kerShiftTetIdx` (KerDivision.cu:1105-1125)
    - Shifts tet indices >= start by shift amount
    - Handles negative indices

### Utility (1 kernel)
12. **make_reverse_map.wgsl** - Port of `kerMakeReverseMap` (KerDivision.cu:816-832)
    - Creates reverse mapping from scatter array

---

### Critical Complex Kernels (2 kernels) ✅ COMPLETED
13. **update_opp.wgsl** - Port of `kerUpdateOpp` (KerDivision.cu:561-695) - **VERY CRITICAL**
    - Updates adjacency after flips (both 2-3 and 3-2)
    - Handles concurrent flip detection via tetMsgArr
    - Implements Flip32NewFaceVi lookup table for 3-2 flips
    - Complex logic with external opp extraction and internal opp setup
    - Manages both neighbor-not-flipped and neighbor-also-flipped cases

14. **mark_rejected_flips.wgsl** - Port of `kerMarkRejectedFlips` (KerDivision.cu:195-308)
    - Validates flip votes (checks if all participating tets agree)
    - Uses workgroup shared memory and atomic compaction
    - Checks vote consistency for 2-3 flips (bot + top) and 3-2 flips (bot + top + side)
    - Marks invalid flips as -1 and valid tets as Checked

---

## ⬜ Not Yet Implemented (Requires Exact Predicates)

1. **kerRelocatePointsFast** (KerPredicates.cu:820-934)
   - Relocates vertices through flip graph using fast predicates
   - Requires: doOrient3DFast, flip trace infrastructure

2. **kerRelocatePointsExact** (KerPredicates.cu:936-951)
   - Relocates vertices through flip graph using exact/SoS predicates
   - Requires: doOrient3DSoS (Shewchuk's exact arithmetic)

---

## 📋 Next Steps

### Phase 1: Integrate Completed Shaders
For each of the 14 completed shaders:
1. Add pipeline definition to `src/gpu/pipelines.rs`
2. Add bind group layout and creation
3. Add dispatch function to `src/gpu/dispatch.rs`
4. Identify appropriate call sites (likely in `src/phase1.rs` or new compaction module)

### Phase 2: Test Integration ✅ READY
All critical kernels now implemented! Next: integrate into pipeline and test.

### Phase 3: Exact Predicates (Long-term)
- Port Shewchuk's exact arithmetic (orient3d, insphere)
- Implement SoS (Simulation of Simplicity) tie-breaking
- Port relocation kernels once predicates available

---

## 🔧 Implementation Notes

### TetOpp Encoding Consistency
All shaders use: `(tet_idx << 5u) | opp_vi`
- Extract tet: `opp_val >> 5u`
- Extract vi: `opp_val & 31u`

### Position-Based Indexing
- `vert_tet[position]` NOT `vert_tet[vertex_id]`
- Position = index in uninserted array

### Constants
- `MEAN_VERTEX_DEGREE = 8` (not 64!)
- `TET_ALIVE = 1u` (bit 0)
- `TET_CHANGED = 2u` (bit 1)
- `TET_EMPTY = 4u` (bit 2)
- `OPP_SPECIAL = 8u` (bit 3 in TetOpp)

### Invalid Markers
- `0xFFFFFFFFu` for invalid tet/adjacency
- `-1` in CUDA → `0xFFFFFFFFu` in WGSL (unsigned)
