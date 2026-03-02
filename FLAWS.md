# gDel3D WGPU Port - Unfaithfulness Issues

This document catalogs all unfaithfulness issues found by systematic comparison between the CUDA original (gDel3D) and the WGPU port.

**Status**: Work in Progress - Issues to be addressed one by one

---

## 🔴 CRITICAL: Missing Kernels (Blocks Correctness)

### 1. kerNegateInsertedVerts
- **CUDA Location**: KerDivision.cu lines 872-886
- **Called**: GpuDelaunay.cu line 980 (after pick winner)
- **Purpose**: Marks successfully inserted vertices by negating their vert_tet values
- **Logic**: `if (tetToVert[tetIdx] == idx) vertTetArr[idx] = makeNegative(tetIdx);`
- **Impact**: Without this, inserted vertices can't be distinguished from uninserted ones
- **Status**: ✅ IMPLEMENTED (2026-03-01) - negate_inserted_verts.wgsl

### 2. kerMarkTetEmpty
- **CUDA Location**: KerDivision.cu lines 787-795
- **Called**: GpuDelaunay.cu line 1013 (before split)
- **Purpose**: Sets empty flag (bit 2) on all tets before split
- **Logic**: `setTetEmptyState(tetInfoVec._arr[idx], true);` - sets bit 2
- **Impact**: Empty flag affects flip behavior (empty tets produce invalid flips)
- **Status**: ✅ IMPLEMENTED (2026-03-01) - mark_tet_empty.wgsl

### 3. kerMarkRejectedFlips
- **CUDA Location**: KerDivision.cu lines 195-308
- **Called**: GpuDelaunay.cu line 1154 (during flipping)
- **Purpose**: Validates flip votes by checking bilateral consensus
- **Logic**: Checks bottom tet, top tet, and (for 3-2) side tet all voted consistently
- **Impact**: Invalid flips may execute without validation
- **Status**: ❌ NOT IMPLEMENTED

### 4. kerUpdateOpp
- **CUDA Location**: KerDivision.cu lines 561-695
- **Called**: GpuDelaunay.cu line 1289 (after flip execution)
- **Purpose**: Updates adjacency after flips using encoded face info
- **Logic**: Decodes tetMsgArr and updates all external neighbor pointers
- **Impact**: Adjacency corruption after flips
- **Status**: ❌ NOT IMPLEMENTED (partially in flip.wgsl but different approach)

### 5. kerCompactTets
- **CUDA Location**: KerDivision.cu lines 1191-1234
- **Called**: GpuDelaunay.cu line 1396 (periodic cleanup)
- **Purpose**: Compacts tet arrays, removes dead tets, remaps adjacency
- **Logic**: Moves alive tets to new positions, updates all opp pointers
- **Impact**: Dead tets accumulate, unbounded memory growth
- **Status**: ❌ NOT IMPLEMENTED

### 6. kerMakeCompactMap
- **CUDA Location**: KerDivision.cu lines 1171-1189
- **Called**: GpuDelaunay.cu line 1387 (before compaction)
- **Purpose**: Builds mapping from old tet indices to new compacted positions
- **Impact**: Required for kerCompactTets
- **Status**: ❌ NOT IMPLEMENTED

### 7. kerCollectFreeSlots
- **CUDA Location**: KerDivision.cu lines 1152-1169
- **Called**: GpuDelaunay.cu line 1378 (before compaction)
- **Purpose**: Collects indices of dead tets for reuse
- **Impact**: Can't reuse dead tet slots
- **Status**: ❌ NOT IMPLEMENTED

### 8. kerMakeReverseMap
- **CUDA Location**: KerDivision.cu lines 816-832
- **Called**: GpuDelaunay.cu line 612 (during vertex reordering)
- **Purpose**: Creates reverse mapping for inserted vertices after sorting
- **Impact**: Needed for block-based allocation with sorting
- **Status**: ❌ NOT IMPLEMENTED

### 9. kerUpdateFlipTrace
- **CUDA Location**: KerDivision.cu lines 741-785
- **Called**: GpuDelaunay.cu line 1435 (in relocateAll)
- **Purpose**: Builds flip history chain for point relocation
- **Impact**: Points can't be relocated through flip graph
- **Status**: ❌ NOT IMPLEMENTED

### 10. kerRelocatePointsFast/Exact
- **CUDA Location**: KerPredicates.cu lines 822-917, 920-951
- **Called**: GpuDelaunay.cu lines 1447, 1454 (after flipping)
- **Purpose**: Walks through flip graph to relocate vertices after flips
- **Logic**: Uses tetToFlip mapping and orient3d to navigate new tets
- **Impact**: Vertices with dead tets aren't properly relocated
- **Status**: ⚠️ PARTIAL (update.wgsl has simple invalidation, not full relocation)

### 11. kerMarkSpecialTets
- **CUDA Location**: KerDivision.cu lines 834-870
- **Called**: GpuDelaunay.cu lines 900, 915 (during flipping loop)
- **Purpose**: Clears special adjacency markers
- **Impact**: Affects flip convergence
- **Status**: ❌ NOT IMPLEMENTED

### 12. kerUpdateBlockVertFreeList
- **CUDA Location**: KerDivision.cu lines 956-990
- **Called**: GpuDelaunay.cu line 658 (during block reordering)
- **Purpose**: Updates free list after vertex block reordering
- **Impact**: Free list corruption after reordering
- **Status**: ❌ NOT IMPLEMENTED

### 13. kerUpdateBlockOppTetIdx
- **CUDA Location**: KerDivision.cu lines 1015-1048
- **Called**: GpuDelaunay.cu line 621 (during block reordering)
- **Purpose**: Remaps opp tet indices after block reordering
- **Impact**: Adjacency corruption after reordering
- **Status**: ❌ NOT IMPLEMENTED

### 14. kerShiftInfFreeIdx, kerShiftOppTetIdx, kerShiftTetIdx, kerUpdateTetIdx
- **CUDA Location**: KerDivision.cu lines 992-1125
- **Called**: Various places during buffer growth/reordering
- **Purpose**: Index shifting operations during compaction/growth
- **Impact**: Index corruption during memory operations
- **Status**: ❌ NOT IMPLEMENTED

---

## 🟠 HIGH PRIORITY: Incorrect Implementations

### 1. vote.wgsl: Dummy Insphere Implementation
- **File**: `src/shaders/vote.wgsl` lines 30-34
- **Issue**: `in_sphere_det()` returns `dist_to_centroid()` instead of actual insphere determinant
- **CUDA**: Uses `dPredWrapper.inSphereDet(tet, vert)` (actual determinant)
- **Impact**: Affects insertion point selection ordering
- **Fix**: Implemented full insphere determinant calculation matching CUDA insphereDet
- **Status**: ✅ FIXED (2026-03-01)

### 2. No Exact Predicate Variants
- **Issue**: WGSL only has Fast versions, no Exact fallback
- **CUDA Has**:
  - `kerInitPointLocationFast/Exact`
  - `kerSplitPointsFast/Exact`
  - `kerRelocatePointsFast/Exact`
  - `doOrient3DFast/SoS`
  - `doInSphereFast/SoS`
- **WGSL Has**: Only simple fast versions
- **Impact**: No robustness for degenerate cases
- **Status**: ⚠️ ARCHITECTURAL DIFFERENCE

### 3. No SoS (Simulation of Simplicity) Tiebreaker
- **CUDA**: Full 14-depth SoS decision tree for orient3d, 49-depth for insphere
- **WGSL**: None - relies entirely on double-double precision
- **Impact**: Fails on coplanar points, cocircular vertices, edge midpoints
- **Correctness**: Breaks on non-general-position inputs
- **Status**: ❌ MISSING ROBUSTNESS

---

## 🟡 MEDIUM PRIORITY: Missing Constants & Lookup Tables

### 1. Flip32NewFaceVi
- **CUDA Location**: GPUDecl.h lines 149-153
- **Definition**: `const int Flip32NewFaceVi[3][2]`
- **Purpose**: Vertex index mapping for 3-2 flips
- **Needed By**: flip.wgsl (if 3-2 flips are implemented)
- **Status**: ❌ NOT PORTED

### 2. Flip23IntFaceOpp
- **CUDA Location**: GPUDecl.h lines 155-159
- **Definition**: `const int Flip23IntFaceOpp[3][4]`
- **Purpose**: Internal face adjacency for 2-3 flips
- **Needed By**: flip.wgsl
- **Status**: ❌ NOT PORTED

### 3. TetNextViAsSeenFrom
- **CUDA Location**: CommonTypes.h lines 135-140
- **Definition**: `const int TetNextViAsSeenFrom[4][4]`
- **Purpose**: Successor ordering for tet vertices
- **Usage**: May not be needed in current port
- **Status**: ❌ NOT PORTED (possibly not needed)

---

## 🔵 ARCHITECTURAL DIFFERENCES

### 1. Two-Phase Fast+Exact Strategy Missing
- **CUDA**:
  - Phase 1: Fast predicates, mark uncertain points (orient == 0)
  - Phase 2: Exact predicates with SoS for marked points
- **WGSL**: Single-phase fast predicates only
- **Impact**: Cannot selectively apply expensive exact computation
- **Deferred Processing**: CUDA marks points with `makeNegative(tetIdx)`, WGSL has no equivalent
- **Status**: Fundamental architectural difference

### 2. Point Location Algorithm
- **CUDA**: Decision tree through super-tet (4 fixed steps) for initialization
- **WGSL**: Stochastic walk (max 512 steps) for all iterations
- **Impact**: Different convergence behavior
- **Assessment**: WGSL approach is actually more general-purpose
- **Status**: ✅ INTENTIONAL IMPROVEMENT

### 3. Flip Detection & Execution
- **CUDA**: 3-step pipeline (Check → Reject → Flip → UpdateOpp)
- **WGSL**: Unified kernel with CAS locking (Check+Flip+UpdateOpp in one)
- **Impact**: Different concurrency control mechanism
- **Assessment**: Both valid, different trade-offs
- **Status**: ⚠️ ARCHITECTURAL CHOICE

### 4. 3-2 Flip Support Missing
- **CUDA**: Handles both Flip23 (2→3 tets) and Flip32 (3→2 tets)
- **WGSL**: Only Flip23 implemented in flip.wgsl
- **Impact**: Cannot reduce tet count via 3-2 flips
- **Status**: ❌ INCOMPLETE

### 5. Infinity Point Handling
- **CUDA**: Explicit infinity vertex with special predicate handling
- **WGSL**: Finite point set, super-tet manages boundary implicitly
- **Assessment**: Appropriate for WGSL's approach
- **Status**: ✅ ACCEPTABLE DIFFERENCE

---

## 🟢 VERIFIED CORRECT

### 1. TetOpp Encoding
- **Encoding**: `(tet_idx << 5u) | opp_vi` ✅
- **Decode Tet**: `>> 5u` ✅
- **Decode Vi**: `& 31u` (for full field) or `& 3u` (for 2-bit vi only) ✅
- **Status**: ✅ CORRECT in all shaders

### 2. Position-Based vert_tet Indexing
- **Semantics**: `vert_tet[position]` not `vert_tet[vertex]` ✅
- **Usage**: Consistently position-indexed throughout ✅
- **Status**: ✅ CORRECT (fixed 2026-02-28)

### 3. tet_to_vert Semantics
- **CUDA**: Stores position in uninserted array
- **WGSL**: Now correctly stores position (fixed from insertion index)
- **Status**: ✅ FIXED (2026-03-01)

### 4. Neighbor Split Detection
- **Logic**: Checks if `tet_to_vert[nei_tet] == INT_MAX` ✅
- **Free Array Lookup**: `free_arr[(split_vert+1)*MEAN_VERTEX_DEGREE-1 - nei_vi]` ✅
- **Status**: ✅ CORRECT in split.wgsl

### 5. Core Constants
- **TET_VI_AS_SEEN_FROM**: ✅ Exact match
- **INT_SPLIT_FACE_OPP**: ✅ Exact match
- **SPLIT_FACES**: ✅ Exact match
- **SPLIT_NEXT**: ✅ Exact match
- **MEAN_VERTEX_DEGREE**: ✅ Correct (8, not 64 as one agent erroneously reported)
- **Status**: ✅ ALL CORRECT

### 6. Buffer Sizing
- **All buffer sizes**: ✅ Match CUDA allocations
- **Counter initialization**: ✅ Correct
- **Super-tet initialization**: ✅ Exact match (5 tets)
- **Status**: ✅ CORRECT

---

## 📋 SUMMARY STATISTICS

| Category | Count | Status |
|----------|-------|--------|
| **Missing Kernels** | 12 (2 implemented) | ⚠️ Critical |
| **Incorrect Implementations** | 2 (1 fixed) | ⚠️ High Priority |
| **Missing Constants** | 3 | ⚠️ Medium Priority |
| **Architectural Differences** | 5 | ⚠️ Design Choices |
| **Verified Correct** | 6 categories | ✅ Working |

**Total Issues**: 25 unfaithfulness problems identified (3 resolved)
**Critical Blockers**: 12 missing kernels + 2 incorrect implementations = 14 issues
**Non-Critical**: 8 issues (architectural differences or minor missing pieces)
**Resolved**: 2 kernels (kerMarkTetEmpty, kerNegateInsertedVerts) + 1 fix (insphere in vote.wgsl)

---

## 🎯 RECOMMENDED FIX ORDER

### Phase 1: Critical Missing Kernels (Enables Basic Functionality)
1. ✅ Fix tet_to_vert semantics (DONE 2026-03-01)
2. ✅ Implement kerMarkTetEmpty (DONE 2026-03-01)
3. ✅ Implement kerNegateInsertedVerts (DONE 2026-03-01)
4. ✅ Fix vote.wgsl in_sphere_det() to use real determinant (DONE 2026-03-01)
5. ⬜ Test basic 4-point insertion - **Still segfaulting after all Phase 1 fixes**

### Phase 2: Flip Support (Enables Delaunay Refinement)
6. ⬜ Implement kerMarkRejectedFlips (or verify CAS locking suffices)
7. ⬜ Verify kerUpdateOpp equivalent in flip.wgsl
8. ⬜ Add Flip32NewFaceVi and Flip23IntFaceOpp constants
9. ⬜ Implement 3-2 flip support
10. ⬜ Test flipping on simple cases

### Phase 3: Memory Management (Prevents Unbounded Growth)
11. ⬜ Implement prefix sum for compaction
12. ⬜ Implement kerCollectFreeSlots
13. ⬜ Implement kerMakeCompactMap
14. ⬜ Implement kerCompactTets
15. ⬜ Test long-running insertions

### Phase 4: Advanced Features (Block Reordering, Relocation)
16. ⬜ Implement kerMakeReverseMap
17. ⬜ Implement block update kernels (kerUpdateBlock*)
18. ⬜ Implement full kerRelocatePoints
19. ⬜ Implement kerUpdateFlipTrace
20. ⬜ Test complex point sets

### Phase 5: Robustness (Optional, For Production)
21. ⬜ Add SoS tiebreaker for orient3d
22. ⬜ Add SoS tiebreaker for insphere
23. ⬜ Implement Exact predicate variants
24. ⬜ Add two-phase Fast→Exact pipeline
25. ⬜ Test degenerate cases (coplanar, cocircular)

---

## 📝 NOTES

- This analysis was performed by 12 parallel agents systematically comparing the entire CUDA codebase against the WGSL port
- Agent reports dated: 2026-03-01
- Original CUDA source: `/home/amai/gDel3D/src/gdel3d/gDel3D/`
- WGSL port: `/home/amai/gdel3d_wgpu/`

### Known False Positives
- One agent incorrectly reported MEAN_VERTEX_DEGREE as 64 in CUDA; it is actually 8 ✅
- Some agents reported `& 31u` as wrong for vi decoding; it's correct for extracting from full opp field ✅

### Key Insights
1. The current segfault is likely caused by missing kerMarkTetEmpty or kerNegateInsertedVerts
2. The port cannot work correctly without compaction kernels (unbounded memory growth)
3. The architectural differences (single-phase predicates, CAS locking) are acceptable design choices
4. SoS robustness is optional for general-position inputs but required for production use

---

**Last Updated**: 2026-03-01
**Status**: Ready to begin systematic fixes
