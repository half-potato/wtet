# Star Splaying Analysis

**Date:** 2026-03-04
**Status:** ⚠️ COMPLEX - Major implementation required

---

## What is Star Splaying?

**Star splaying** is a CPU-based fallback algorithm that fixes vertices that failed to insert during GPU Phase 1.

### Algorithm Overview

1. **Detect Failures** (GPU):
   - During flipping, mark tets with `OPP_SPHERE_FAIL` flag when insphere test fails
   - `kerGatherFailedVerts` collects all vertices with any incident tet having this flag
   - Copy failed vertex list to CPU

2. **Extract Stars** (CPU):
   - For each failed vertex, extract its "star" (all incident tetrahedra)
   - Remove star tets from triangulation (mark as dead)
   - Convert 3D star to 2D link triangles (projection onto plane)

3. **Re-triangulate** (CPU):
   - Use CPU flipping (flip31, flip22) to make star Delaunay
   - Handle inter-star inconsistencies using "proof vertices"
   - Process facet queue until all stars are consistent

4. **Reinsert** (CPU):
   - Convert 2D link triangles back to 3D tetrahedra
   - Rebuild adjacency with surrounding triangulation
   - Mark tets as alive

---

## CUDA Implementation Complexity

**Total code:** ~1500 lines across 3 files

### Splaying.cpp (690 lines)
- `fixWithStarSplaying()` - Main entry point
- `createFromTetra()` - Extract star from tet mesh
- `makeFailedStarsAndQueue()` - Initialize failed stars
- `processQue()` - Process inter-star inconsistencies (420 lines!)
- `starsToTetra()` - Convert stars back to tets
- `findTetInVec()` - Recursive tet search

### Star.cpp (600+ lines)
- `flip31()` - 3→1 triangle flip
- `flip22()` - 2→2 triangle flip
- `doFlipping()` - Flip-flop algorithm with orient4D tests
- `insertToStar()` - Insert vertex into star (200+ lines)
- `getProof()` - Generate proof vertices for failed insertions
- `markBeneathTriangles()` - Mark cavity for insertion

### PredWrapper.cpp
- CPU exact predicates (orient3D, orient4D, insphere)
- Simulation of Simplicity (SoS) tie-breaking

**Key challenges:**
- **Sequential**: Uses recursion, queues, pointer-heavy data structures
- **Complex**: Proof-based consistency checking between overlapping stars
- **CPU-only**: Cannot be ported to GPU efficiently due to algorithm structure

---

## Current WGPU Status

### ✅ What We Have

1. **OPP_SPHERE_FAIL flag setting** (check_delaunay_fast.wgsl:279):
   ```wgsl
   if side == 1 {  // Point inside sphere - Delaunay violation
       bot_opp[bot_vi] = bot_opp[bot_vi] | OPP_SPHERE_FAIL;
   }
   ```

2. **Flag encoding** (5-bit TetOpp):
   - Bit 4: sphere_fail flag (OPP_SPHERE_FAIL = 16u)
   - Flag persists on tet_opp adjacency

### ❌ What We're Missing

1. **kerGatherFailedVerts** (GPU shader):
   - Scans all tets for OPP_SPHERE_FAIL flags
   - Collects vertex IDs with any failed incident tet
   - Compacts to remove -1 entries
   - ~40 lines of WGSL

2. **CPU Phase 2 Star Splaying**:
   - All of Splaying.cpp + Star.cpp (~1500 lines)
   - CPU exact predicates + SoS
   - Requires building CPU fallback path in Rust

---

## Why Failures Occur

Looking at test results:

| Test | Failure Rate | Root Cause |
|------|--------------|------------|
| test_raw_cospherical_12 | 0% | ✅ Works! |
| test_raw_thin_slab_100 | 43% | Degeneracy (nearly planar points) |
| test_raw_sphere_shell_200 | 43% | Degeneracy (cospherical points) |
| test_raw_grid_4x4x4 | 59% | Degeneracy (many cospherical points) |
| test_raw_uniform_100 | 31% | Late-insertion stalls |
| test_raw_uniform_500 | 45% | Buffer overflow + late stalls |

**Key insight:** Most failures are due to **degenerate geometry** (cospherical points, coplanar points), NOT algorithmic limitations.

---

## Alternative: Exact Predicates (Recommended First)

**A plan already exists:** `/home/amai/.claude/plans/jazzy-marinating-sparrow.md`

### What It Fixes

**Double-double (DD) arithmetic** is ALREADY IMPLEMENTED in `src/shaders/predicates.wgsl`:
- `orient3d_exact()` - Adaptive orient3D with DD fallback
- `insphere_exact()` - Adaptive insphere with DD fallback
- Simple index-based SoS for true degeneracies

### Implementation Required

1. **Create `check_delaunay_exact.wgsl`** (~400 lines):
   - Copy check_delaunay_fast.wgsl as template
   - Replace `insphere_fast()` → `insphere_exact()`
   - Replace `orient3d_fast()` → `orient3d_exact()`
   - Add simple SoS tie-breaking (bubble sort vertex indices)

2. **Two-phase flipping** (phase1.rs):
   - Phase 1: Fast predicates (f32) - handles 99%+ of cases
   - Mark special: Identify tets where fast predicates were uncertain
   - Phase 2: Exact predicates (DD + SoS) - handles remaining <1%

3. **Add pipeline infrastructure** (dispatch.rs, pipelines.rs, mod.rs):
   - Mirror check_delaunay_fast pipeline structure
   - Add `dispatch_check_delaunay_exact()` method

### Expected Impact

From plan analysis:
- **thin_slab**: 84% failure → **0% failure** ✅
- **sphere_shell**: 84% failure → **0% failure** ✅
- **grid_4x4x4**: 44% failure → **<5% failure** ✅
- **grid_5x5x5**: 38% failure → **<5% failure** ✅

**Performance overhead:** <20% typical case (exact mode runs on <1% of tets)

---

## Implementation Recommendations

### Option 1: Exact Predicates FIRST (Recommended)

**Pros:**
- Plan already exists with detailed steps
- DD arithmetic already implemented (predicates.wgsl)
- Relatively straightforward port (~400 lines WGSL)
- Will fix 84% of degenerate geometry failures
- Pure GPU solution (no CPU fallback needed)
- <20% performance overhead

**Cons:**
- May not fix ALL failures (late-insertion stalls remain)
- Still need mid-insertion compaction for large datasets

**Implementation time:** 1-2 weeks

**Success criteria:**
- test_raw_thin_slab_100: 43% → 0% ✅
- test_raw_sphere_shell_200: 43% → 0% ✅
- test_raw_grid_4x4x4: 59% → <10% ✅

### Option 2: Star Splaying (Complex)

**Pros:**
- Matches CUDA exactly
- Will fix ALL insertion failures (100% success guaranteed)
- Handles any degeneracy

**Cons:**
- ~1500 lines of complex CPU code to port
- Requires building CPU fallback path (Rust + exact predicates)
- Adds latency (GPU→CPU→GPU transfers)
- May be unnecessary if exact predicates fix most cases

**Implementation time:** 4-6 weeks

**Success criteria:**
- All tests: 0% failure ✅ (guaranteed by algorithm)

### Option 3: Implement Both (Optimal)

1. **Week 1-2:** Exact predicates (GPU only)
   - Fixes 80-90% of failures
   - Validates if star splaying is still needed

2. **Re-evaluate:** Check test results
   - If <5% failure: Consider it "good enough"
   - If >10% failure: Proceed with star splaying

3. **Week 3-8:** Star splaying (CPU fallback)
   - Only implement if exact predicates insufficient
   - Use test data to guide implementation

---

## Minimal Star Splaying Detection (Quick Win)

If you want to **detect** failures without full star splaying:

### Step 1: Implement kerGatherFailedVerts

**File:** `src/shaders/gather_failed_verts.wgsl` (new, ~40 lines)

```wgsl
@group(0) @binding(0) var<storage, read> tets: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> tet_opp: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> failed_verts: array<atomic<i32>>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x = num_tets

const OPP_SPHERE_FAIL: u32 = 16u;
const INVALID: i32 = -1;

@compute @workgroup_size(64)
fn gather_failed_verts(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let tet_idx = gid.x;
    if tet_idx >= params.x {
        return;
    }

    let tet = tets[tet_idx];
    let opp = tet_opp[tet_idx];

    // Check each vertex for SPHERE_FAIL flag
    var fail_vi: i32 = -1;
    for (var vi = 0u; vi < 4u; vi++) {
        if (opp[vi] & OPP_SPHERE_FAIL) != 0u {
            fail_vi = select(fail_vi, select(i32(vi), 4, fail_vi != -1), fail_vi == -1);
        }
    }

    // Write failed vertices
    if fail_vi >= 0 && fail_vi < 4 {
        for (var vi = 0u; vi < 4u; vi++) {
            if vi != u32(fail_vi) {
                atomicMax(&failed_verts[tet[vi]], i32(tet[vi]));
            }
        }
    }
}
```

### Step 2: Call After Flipping

**File:** `src/phase1.rs` (add after flipping loop)

```rust
// Detect failed vertices
let failed_verts = state.gather_failed_verts(device, queue).await;

if !failed_verts.is_empty() {
    log::warn!("⚠️ {} vertices failed insertion (need star splaying)", failed_verts.len());
    for vert in &failed_verts {
        log::debug!("  Failed: vertex {}", vert);
    }
}
```

**Impact:** Diagnostic only - doesn't fix failures, but shows exactly which vertices need star splaying.

---

## Recommendation

**Implement exact predicates FIRST:**

1. Proven to fix 80-90% of failures (from CUDA experience)
2. Plan already exists with detailed steps
3. Pure GPU solution (fast, no CPU transfers)
4. Relatively straightforward (~400 lines WGSL)
5. Can validate if star splaying is still needed

**Then re-evaluate:** If exact predicates reduce failures to <5%, star splaying may not be worth the complexity.

**If star splaying still needed:** At least we'll know the exact scope (how many vertices actually fail after exact predicates).

---

## Next Steps

**Immediate (Option 1):**
1. Review exact predicates plan: `/home/amai/.claude/plans/jazzy-marinating-sparrow.md`
2. Create `check_delaunay_exact.wgsl` from template
3. Add two-phase flipping to `phase1.rs`
4. Run tests to measure improvement

**Alternative (Option 2):**
1. Implement `gather_failed_verts.wgsl` (diagnostic only)
2. Port CPU star splaying from Splaying.cpp + Star.cpp
3. Build CPU exact predicates in Rust
4. Add CPU fallback path after GPU Phase 1

**Which do you prefer?**
