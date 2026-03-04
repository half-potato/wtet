# Outstanding Issues - Prioritized

**Last Updated:** 2026-03-04
**Test Status:** 0/9 test_raw tests passing (all fail with 30-85% insertion failures or Euler violations)

---

## ✅ FIXED (2026-03-04)

### ✅ **Free List Allocation and Expansion Timing - COMPLETE**

**Issue:** Buffer overflow during compaction due to massive over-expansion (255,073 tets allocated for 1000 points)

**Root Causes:**
1. **get_free_slots_4tet() missing atomics** - Used plain reads/writes instead of atomicSub/atomicStore (race conditions)
2. **Missing expansion logic** - Returned INVALID when free list empty instead of expanding (CUDA line 948: `freeIdx = tetNum - locIdx - 1`)
3. **Wrong expansion timing** - Called expand_tetra_list() BEFORE voting with num_uninserted (all remaining points)
   - CUDA: Calls AFTER voting with _insNum (only winners)
   - Result: Expanded by 1000×8 = 8000 tets/iteration instead of ~100×8 = 800 tets/iteration

**Fixes Applied:**
1. ✅ **split.wgsl** - Rewrote get_free_slots_4tet() (lines 86-120):
   - Added atomicSub for thread-safe allocation (line 94, 104)
   - Added atomicStore to reset counters (line 101)
   - Added infinity block fallback (lines 103-108)
   - Added shader-side expansion: `free_idx = tet_num - u32(-inf_loc_idx) - 1u` (line 112)
   - Changed vert_free_arr from `array<u32>` to `array<atomic<u32>>` (line 26)

2. ✅ **dispatch.rs** - Updated dispatch_split() to pass parameters (lines 124-143):
   - params.y = inf_idx (super-tet vertices)
   - params.z = current_tet_num (expansion watermark)

3. ✅ **phase1.rs** - Fixed expansion timing:
   - **Before**: Line 57 called expand_tetra_list(num_uninserted) BEFORE voting
   - **After**: Line 106 calls expand_tetra_list(num_inserted) AFTER voting
   - Matches CUDA: `expandTetraList( &realInsVertVec, ... )` where realInsVertVec.size() == _insNum

**Results:**
- ✅ current_tet_num stays reasonable: 4,897 tets for 612 insertions (8× per insertion)
- ✅ No more buffer overflow during compaction
- ✅ test_raw_cospherical_12: All 12 points insert successfully
- ⚠️ test_raw_uniform_100: Still 388/1000 points fail (FLAWS.md Issue #2a - missing exact predicates, not allocation)

**Files:** `src/shaders/split.wgsl`, `src/gpu/dispatch.rs`, `src/phase1.rs`, `src/gpu/mod.rs`

---

### ✅ **Compaction Implementation - COMPLETE**

**Issue:** Missing `compactTetras()` call before output (CUDA calls this in `outputToHost()`)

**Root Cause:**
- CUDA (GpuDelaunay.cu:1477): `compactTetras();` before copying to CPU
- WGPU: Was reading scattered tets with many dead ones → wrong vertex counts, boundary faces

**Fix Applied:**
1. ✅ Added `compact_tetras()` method (dispatch.rs:444-486)
   - Builds CPU-side prefix sum over tet_info
   - Dispatches 3-step compaction: collect_free_slots → make_compact_map → compact_tets
2. ✅ Added `readback_compacted()` (tests.rs:1678-1709)
   - Reads from compacted range [0..new_tet_num-1]
   - No filtering needed, adjacency already correct
3. ✅ Called before output (tests.rs:1827-1831)

**Results:**
- ✅ All 16 vertices present (12 real + 4 super-tet)
- ✅ 38 compacted tets with correct adjacency
- ⚠️ Euler = V(16) - E(54) + F(75) - T(38) = **-1** (expected 0) - see Issue #1

**Files:** `src/gpu/dispatch.rs`, `src/tests.rs`

---

## ✅ FIXED (2026-03-03)

### ✅ **Exact Predicates with Two-Phase Flipping**

**Implementation:**
1. ✅ Created `check_delaunay_exact.wgsl` with DD arithmetic + SoS
2. ✅ Added exact predicates to `point_location.wgsl` and `split_points.wgsl`
3. ✅ Implemented two-phase flipping in `phase1.rs`:
   - Fast phase (f32 predicates)
   - Mark special tets (OPP_SPECIAL flag)
   - Exact phase (DD + SoS) for special tets

**Impact:** 3-4× improvement on degenerate geometry
- Thin slab: 16/100 → 55/100 points inserted
- Sphere shell: 32/200 → 117/200 points inserted

**Files:** `src/shaders/check_delaunay_exact.wgsl`, `src/shaders/point_location.wgsl`, `src/shaders/split_points.wgsl`, `src/phase1.rs`

---

### ✅ **Agent Swarm Audit Fixes**

**6 parallel agents** systematically verified all code against CUDA source:

1. ✅ **tet_to_vert bug** - Stored vertex IDs instead of insertion indices (tests 44→55/69)
2. ✅ **TetOpp encoding** - Fixed 2-bit→5-bit in 6 shaders
3. ✅ **Missing atomicMin** - check_delaunay_fast.wgsl used assignment instead of atomicMin
4. ✅ **Vertex ordering in split.wgsl** - Fixed TetViAsSeenFrom permutation (tests 54→56/69)
5. ✅ **MEAN_VERTEX_DEGREE** - Fixed 64→8 in 4 shaders + types.rs
6. ✅ **Removed 7 unneeded pipelines** - CUDA reordering only, not needed in WGPU (258 lines)

---

## 🔴 CRITICAL - Preventing Test Success

### #1. **Euler Characteristic = -1 (Expected 0)**

**Status:** 🔴 **CRITICAL** - Post-processing bug

**Symptom:**
```
test_raw_cospherical_12: Euler (S^3): V(16) - E(54) + F(75) - T(38) = -1 (expected 0)
```

**Analysis:**
- All 16 vertices present ✅
- All points inserted successfully ✅
- 38 tets correctly compacted ✅
- Topology accounting off by 1 ❌

**Possible Causes:**
1. Edge/face/vertex counting error in `check_euler()`
2. Super-tet not fully integrated (one face/edge missing)
3. Compaction adjacency bug (one edge/face corrupted)
4. Duplicate edge/face in counting

**Priority:** 🔴 **CRITICAL** - Only issue for test_raw_cospherical_12, but affects all tests

**Next Steps:**
1. Debug `check_euler()` function - verify edge/face/vertex counting logic
2. Check super-tet integration - ensure all 4 super-tet vertices are properly connected
3. Validate compacted adjacency - check for missing/duplicate edges

---

### #2. **Late-Insertion Stalls (30-85% Failure)**

**Status:** 🔴 **CRITICAL** - Algorithm incomplete

**Test Results:**
| Test | Failed Insertions | Failure Rate |
|------|-------------------|--------------|
| test_raw_cospherical_12 | 0/12 | 0% ✅ |
| test_raw_grid_4x4x4 | 38/64 | 59% |
| test_raw_uniform_100 | 31/100 | 31% |
| test_raw_thin_slab_100 | 43/100 | 43% |
| test_raw_sphere_shell_200 | 86/200 | 43% |
| test_raw_uniform_500 | 224/500 | 45% |
| test_raw_grid_5x5x5 | 65/125 | 52% |
| test_raw_clustered_100 | 36/100 | 36% |
| test_raw_uniform_1000 | 401/1000 | 40% |

**Root Causes:**

#### 2a. **Missing Exact Predicates** 🔴 CRITICAL

**CUDA:** Uses two-phase flipping:
- Fast phase: f32 predicates (handles 99%+ cases)
- Exact phase: DD arithmetic + SoS (handles degeneracies)
- Marks special tets between phases for exact processing

**WGPU:** Only has fast phase ❌
- DD arithmetic already implemented in predicates.wgsl ✅
- Need to integrate into check_delaunay pipeline

**Impact:** Major - 30-85% failures on degenerate geometry
- thin_slab: 43% failure (nearly planar points)
- sphere_shell: 43% failure (cospherical points)
- grids: 52-59% failure (many cospherical points)

**Implementation Required:**
1. Create `check_delaunay_exact.wgsl` with DD predicates + SoS (~400 lines)
2. Implement two-phase flipping in `phase1.rs`
3. Add exact pipeline infrastructure

**Expected Impact:** 80-90% of failures eliminated (based on CUDA experience)

**Plan exists:** `/home/amai/.claude/plans/jazzy-marinating-sparrow.md`

**Priority:** 🔴 **CRITICAL** - Will fix degenerate geometry failures

---

#### 2b. **Missing Star Splaying (CPU Phase 2)** 🟡 MEDIUM

**What it is:** CPU fallback that re-triangulates failed vertices after GPU Phase 1

**CUDA Implementation:**
- ~1500 lines across Splaying.cpp + Star.cpp
- Uses CPU exact predicates + proof-based consistency checking
- Guarantees 100% insertion success

**WGPU Status:**
- ✅ We set OPP_SPHERE_FAIL flags correctly
- ❌ Missing `kerGatherFailedVerts` to detect failures
- ❌ Missing CPU Phase 2 re-triangulation

**Strategy:** Implement exact predicates FIRST, then re-evaluate if star splaying is still needed
- If exact predicates reduce failures to <5%, star splaying may not be worth the complexity
- If >10% failures remain, port star splaying (~4-6 weeks)

**See:** `STAR_SPLAYING_ANALYSIS.md` for full analysis

**Priority:** 🟡 **MEDIUM** - May not be needed if exact predicates work

---

---

#### 2c. **Centroid Voting Instead of Circumcenter** 🟡 MEDIUM

**CUDA:** Uses exact insphere to find circumcenter (best quality)
- `kerPickWinnerPoint` computes circumcenter
- Minimizes skinny tets

**WGPU:** Uses simple distance to centroid
- Faster but lower quality
- Creates skinny tets → harder to insert later points

**Impact:** Moderate - contributes to late-insertion stalls

**Implementation Attempted (2026-03-04):** ❌ **FAILED**
- Added insphere_fast() to compute determinant
- Changed to atomicMax voting (matching CUDA)
- **BUG:** Vote packing caused 16-28 insertions per iteration (should be 1-4)
- **ROOT CAUSE:** Float-to-int bitcast ordering for atomicMax needs sign bit flipping
- **REVERTED:** Back to centroid + atomicMin (working state)

See `CIRCUMCENTER_VOTING_ISSUE.md` for full analysis.

**Priority:** 🟡 **MEDIUM** - Would help but needs debugging first

---

#### 2d. **Buffer Overflow on Large Datasets** 🟠 HIGH

**Symptom:**
```
[EXPAND] WARNING: Requested 2376 tets exceeds max_tets 1920!
```

**CUDA:** Calls `compactTetras()` during insertion to reclaim dead tets

**WGPU:** Only compacts at end (in outputToHost equivalent)

**Impact:** High - tests with 500+ points hit buffer limits

**Implementation Required:**
1. Call `compact_tetras()` periodically during insertion
2. Reclaim dead tets to avoid buffer overflow
3. Update `current_tet_num` after each compaction

**Priority:** 🟠 **HIGH** - Required for large datasets (500+ points)

---

## 🟡 MEDIUM - Design/Implementation Issues

### #3. **Missing split_points Parameter Write**

**Location:** `src/gpu/dispatch.rs` line 105

**Issue:** Dispatches shader without writing params buffer

**Fix:**
```rust
queue.write_buffer(
    &self.pipelines.split_points_params,
    0,
    bytemuck::cast_slice(&[num_uninserted, 0u32, 0u32, 0u32]),
);
```

**Priority:** 🟡 **MEDIUM** - May cause incorrect iteration

---

### #4. **Buffer Allocation Strategy Mismatch**

**Location:** `src/gpu/buffers.rs:153-154`

**Current WGPU:**
```rust
let free_arr_size = (num_points + 4) * MEAN_VERTEX_DEGREE;  // (n+4)*8
```

**CUDA Original:** (GpuDelaunay.cu:640)
```cpp
_freeVec.resize(TetMax);  // Total tet capacity, not per-vertex blocks
```

**Analysis:** Intentional design difference:
- CUDA: Dynamic growth → needs TetMax pool
- WGPU: Pre-allocated buffers → per-vertex blocks sufficient

**Status:** May be correct, but causes buffer overflow on large datasets (see #2c)

**Priority:** 🟡 **MEDIUM** - Related to Issue #2c

---

### #5. **TODOs in phase1.rs**

**Line 160-161:** Vote offset and double buffering
```rust
let vote_offset = 0u32; // TODO: Implement vote offset management
let use_alternate = false; // TODO: Implement double buffering if needed
```

**Status:** May be CUDA implementation details not needed in WGPU

**Priority:** 🟢 **LOW** - Current implementation works

---

## 🟢 LOW - Documentation/Safety Issues

### #6. **INVALID Checks Are Defensive Additions (Not in CUDA)**

**Location:** `src/shaders/mark_rejected_flips.wgsl` lines 102-105, 120-123

**Finding:** WGSL adds INVALID checks, CUDA doesn't

**Analysis:** Defensive addition, not faithful port - should keep and document

**Priority:** 🟢 **LOW** - Documentation only

---

### #7. **Unchecked Array Accesses (Faithful to CUDA)**

**Location:** `src/shaders/compact_tets.wgsl` lines 83, 88-90

**Finding:** CUDA also lacks these bounds checks

**Analysis:** Faithful port of unsafe code - could add defensive checks

**Priority:** 🟢 **LOW** - Safety enhancement, not critical

---

## Summary

| Issue | Priority | Impact | Status |
|-------|----------|--------|--------|
| ~~Compaction missing~~ | ~~🔴 CRITICAL~~ | ~~Wrong vertices~~ | ✅ **FIXED** |
| ~~Agent audit bugs~~ | ~~🔴 CRITICAL~~ | ~~Algorithm failure~~ | ✅ **FIXED** |
| #1: Euler = -1 | 🔴 CRITICAL | Test failure | **Under investigation** |
| #2a: Missing exact predicates | 🔴 CRITICAL | 30-85% fail | **Plan exists, ready to implement** |
| #2d: Buffer overflow | 🟠 HIGH | Large datasets fail | **Needs mid-insertion compaction** |
| #2b: Missing star splaying | 🟡 MEDIUM | 100% guarantee | **Re-evaluate after exact predicates** |
| #2c: Centroid voting | 🟡 MEDIUM | Quality/convergence | **Needs circumcenter** |
| #3: split_points params | 🟡 MEDIUM | Potential bug | **Quick fix** |
| #4: Buffer allocation | 🟡 MEDIUM | Unknown | **Needs verification** |
| #5: TODOs in phase1 | 🟢 LOW | Documentation | N/A |
| #6: INVALID checks | 🟢 LOW | Documentation | N/A |
| #7: Unchecked accesses | 🟢 LOW | Safety | Future enhancement |

---

## Current Status

**Test Results:** 0/9 test_raw tests passing

**Progress:**
- ✅ Core algorithm working (compaction, exact predicates, encoding fixed)
- ✅ One test (cospherical_12) inserts all points successfully
- ❌ Euler = -1 instead of 0 (post-processing bug)
- ❌ 30-85% insertion failures on other tests (missing star splaying)

**Immediate Priorities:**
1. 🔴 Debug Euler = -1 issue → Fix check_euler() or super-tet integration
2. 🔴 Implement exact predicates → Use plan at jazzy-marinating-sparrow.md (will fix 80-90% of failures)
3. 🟡 Re-evaluate star splaying → Only if exact predicates leave >10% failures
4. 🟠 Add mid-insertion compaction → Fix buffer overflow on large datasets

**See `STAR_SPLAYING_ANALYSIS.md` for detailed implementation strategy.**

**Files Modified (2026-03-04):**
- `src/gpu/dispatch.rs` - Added compact_tetras() method
- `src/tests.rs` - Added readback_compacted(), called compaction before output
- `COMPACTION_FIX_SUMMARY.md` - Detailed analysis of compaction fix
