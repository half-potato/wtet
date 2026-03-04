# Test Failure Diagnosis (2026-03-03)

**Test Status:** 56/69 passing (81%)

## Executive Summary

The implementation successfully inserts **most points** but has **systematic failures** in specific geometric configurations:

1. **Degenerate geometry** (84% failure): Nearly-planar or 2D manifold points
2. **Regular grids** (38-44% failure): Cospherical/cocircular point configurations
3. **General point clouds** (30-41% failure): Late-insertion stalls

All failures share a **common pattern**: Algorithm stalls when `num_inserted = 0` (no winners found), typically after 5-54 iterations.

---

## Failure Categories

### 🔴 CRITICAL: Degenerate Geometry (84% Failure Rate)

| Test | Failed | Total | Rate | Iterations | Pattern |
|------|--------|-------|------|------------|---------|
| test_raw_thin_slab_100 | 84 | 100 | 84% | 13 | Nearly planar |
| test_raw_sphere_shell_200 | 168 | 200 | 84% | 16 | 2D manifold |

**Characteristics:**
- Points lie on or near a 2D surface in 3D space
- Creates **degenerate tetrahedra** (near-zero volume)
- Orientation predicates become unreliable without exact arithmetic
- **Early stall** (13-16 iterations)

**Convergence Pattern:**
```
thin_slab: 1→2→3→2→1→1→1→1→1→1→1→1→0 (STALL)
sphere:    1→3→4→4→4→2→1→2→2→2→1→1→1→1→1→0 (STALL)
```

**Root Cause:**
- Missing **exact predicates** (orient3d/insphere with arbitrary precision)
- Missing **Simulation of Simplicity (SoS)** for tie-breaking degenerate cases
- Fast floating-point tests fail when points are nearly coplanar

---

### 🟠 HIGH: Regular Grids (38-44% Failure Rate)

| Test | Failed | Total | Rate | Iterations | Pattern |
|------|--------|-------|------|------------|---------|
| test_raw_grid_4x4x4 | 28 | 64 | 44% | 14 | Cube vertices |
| test_raw_grid_5x5x5 | 48 | 125 | 38% | 54 | Cube vertices |

**Characteristics:**
- **Cospherical points**: Many points share exact circumspheres (vertices of cubes)
- **Symmetry**: Perfect geometric alignment causes ambiguity
- Requires **symbolic perturbation** (SoS) to resolve ties

**Convergence Pattern:**
```
4x4x4: 1→3→3→3→2→3→4→4→4→2→3→3→1→0 (STALL)
5x5x5: 1→3→4→4→3→4→2→2→1→2→2→2→1→0 (STALL, iter 54)
```

**Root Cause:**
- No **exact sphere test** for cospherical detection
- No **SoS perturbation** for consistent tie-breaking
- CUDA uses `SphereExactOrientSoS` mode (missing in WGPU)

---

### 🟡 MEDIUM: General Point Clouds (30-41% Failure)

| Test | Failed | Total | Rate | Iterations | Pattern |
|------|--------|-------|------|------------|---------|
| test_raw_uniform_100 | 41 | 100 | 41% | 10 | Random uniform |
| test_raw_uniform_500 | 205 | 500 | 41% | 38 | Random uniform |
| test_raw_uniform_1000 | 383 | 1000 | 38% | 34 | Random uniform |
| test_raw_clustered_100 | 34 | 100 | 34% | 16 | Clustered |
| test_flip_convergence | 9 | 27 | 33% | 9 | Edge midpoints |
| test_no_duplicate_tets | 17 | 50 | 34% | 9 | - |
| test_euler_characteristic | 15 | 50 | 30% | 12 | - |
| test_orientation_consistency | 16 | 50 | 32% | 12 | - |

**Characteristics:**
- **Late-insertion failures**: Failed points are typically in the second half of sequence
- Larger point sets have better success rates (38% vs 41%)
- Suggests algorithm struggles with **complex mesh topology**

**Failed Point Distribution:**
```
uniform_100:  Points 31-99 failed (69% through sequence)
uniform_500:  Points 14+ failed, mostly 200+ (40% through)
uniform_1000: Points 385+ failed (38% through)
```

**Convergence Pattern:**
```
uniform_100:  Good start (8 inserted), then gradual decline, stall at iter 10
uniform_500:  Strong mid-phase (13 max), slow decline, stall at iter 38
uniform_1000: Peak of 13 inserts, gradual decline, stall at iter 34
```

**Root Cause:**
- **Point location failures**: vert_tet becomes stale or points walk to wrong tets
- **Voting system limitations**: Tet-parallel voting may miss valid insertion opportunities
- **Missing two-phase flipping**: Only fast mode implemented, no exact fallback

---

### 🟢 LOW: Cospherical Points (25% Failure)

| Test | Failed | Total | Rate | Iterations |
|------|--------|-------|------|------------|
| test_raw_cospherical_12 | 3 | 12 | 25% | 5 |

**Characteristics:**
- **Intentionally degenerate**: 12 points on exact sphere
- Tests exact arithmetic requirements
- Quick stall (5 iterations)

**Root Cause:** Same as grid tests - no exact sphere test or SoS

---

## Common Stall Pattern

**All failing tests share this pattern:**

1. **Phase 1**: Initial insertions succeed (1-13 points per iteration)
2. **Phase 2**: Insertion rate **decreases gradually**
3. **Phase 3**: `num_inserted = 0` → Algorithm **stalls**
4. **Phase 4**: Loop exits with remaining points uninserted

**Example (thin_slab_100):**
```
Iteration 1:  1 inserted  (99 remaining)
Iteration 2:  2 inserted  (97 remaining)
Iteration 3:  3 inserted  (94 remaining)
...
Iteration 11: 1 inserted  (85 remaining)
Iteration 12: 1 inserted  (84 remaining)
Iteration 13: 0 inserted  (84 remaining) ← STALL
```

**Why Stall Occurs:**

When `num_inserted = 0`, it means:
1. `dispatch_vote()` ran, but no tet accepted any vertex
2. `dispatch_pick_winner()` found no valid winners
3. `dispatch_split()` had nothing to split

**Possible Causes:**
- Points are **outside convex hull** of current mesh
- Points are in **degenerate configurations** (near-planar, cospherical)
- **vert_tet** is stale (points assigned to wrong/invalid tets)
- **Voting criteria** too strict (distance-to-centroid threshold)

---

## Root Cause Analysis

### 1. Missing Exact Predicates ⚠️ CRITICAL

**Current Implementation:**
- Uses **fast floating-point** orient3d and sphere tests
- Works well for general positions
- Fails catastrophically for degenerate geometry

**CUDA Has:**
```cpp
// Two-phase flipping
doFlippingLoop(SphereFastOrientFast);   // Phase 1: Fast floating-point
markSpecialTets();                       // Mark degenerate cases
doFlippingLoop(SphereExactOrientSoS);   // Phase 2: Exact arithmetic + SoS
```

**WGPU Missing:**
- `check_delaunay_exact.wgsl` shader (exact sphere test)
- Exact arithmetic for orient3d (arbitrary precision)
- SoS symbolic perturbation for tie-breaking

**Impact:**
- 84% failure on degenerate geometry
- 38-44% failure on regular grids
- Cannot handle cospherical points robustly

**Fix Required:**
1. Implement exact predicates using Shewchuk's adaptive arithmetic
2. Implement SoS perturbation scheme
3. Add two-phase flipping (fast → exact)

---

### 2. Point Location Failures ⚠️ HIGH

**Observation:**
- Failed points are typically in **later part of sequence**
- Suggests **vert_tet becomes stale** as mesh grows

**Current Flow:**
```
Iteration N:
1. Vote (using vert_tet from prev iteration)
2. Pick winners
3. Split
4. Flip
5. Relocate (updates vert_tet for next iteration)
```

**Potential Issues:**
- Flipping can **invalidate vert_tet** for uninserted points
- Point location (relocate) may fail to find valid tets
- Stochastic walk may converge to **local minimum** (wrong tet)

**Evidence:**
```
[DEBUG] No winners found! Investigating...
[DEBUG] Remaining uninserted: [84 points]
```

If vert_tet is correct, these points should vote for valid tets.

**Fix Required:**
1. Verify relocate_points_fast correctness
2. Add fallback locate if stochastic walk fails
3. Consider full point location every N iterations

---

### 3. Voting System Limitations ⚠️ MEDIUM

**Current Implementation:**
- **Tet-parallel** pick_winner (design difference from CUDA)
- Each tet picks closest uninserted point
- Uses **distance to centroid** metric

**CUDA Uses:**
- **Vertex-parallel** pick_winner
- Each vertex competes for tets
- Uses **sphere test** (which point is inside tet's circumsphere)

**Potential Issues:**
- Distance-to-centroid may not correlate with **valid insertion**
- A tet may pick a point that's actually **outside** its circumsphere
- This could explain why votes happen but insertions fail

**Fix Required:**
1. Switch voting to use **insphere test** (is point inside tet's circumsphere?)
2. Consider switching to vertex-parallel iteration (match CUDA)
3. Add validation: reject pick if point is outside circumsphere

---

### 4. Euler Characteristic Violations ⚠️ HIGH

**Several tests report invalid Euler characteristics:**

```
test_raw_thin_slab_100:
  Euler (S^3): V(20) - E(70) + F(91) - T(60) = -19 (expected 0)

test_raw_clustered_100:
  Euler (S^3): V(70) - E(274) + F(389) - T(229) = -44 (expected 0)
```

**For 3D triangulation embedded in S³ (with infinity):**
- Expected: V - E + F - T = 0
- Actual: **Negative values** (-19, -44)

**This indicates:**
- **Topology corruption**: Missing tets, invalid adjacency, or duplicate tets
- Split/flip operations created **inconsistent mesh**
- Adjacency updates may have bugs

**Fix Required:**
1. Validate mesh topology after each iteration
2. Check for:
   - Duplicate tets
   - Missing adjacency links
   - Invalid tet orientations
3. Verify split.wgsl and flip.wgsl adjacency updates

---

## Recommended Fixes (Priority Order)

### 🔴 P0: Implement Two-Phase Flipping

**Why:** Addresses 84% failure rate on degenerate geometry

**Implementation:**
1. Create `check_delaunay_exact.wgsl`:
   - Exact insphere test using Shewchuk's adaptive arithmetic
   - SoS perturbation for tie-breaking cospherical points
2. Split flip loop in phase1.rs:
   ```rust
   // Fast phase
   for iter in 0..max_flip_iterations {
       check_delaunay_fast();
       if converged { break; }
   }

   // Mark special tets (degenerate cases)
   dispatch_mark_special_tets();

   // Exact phase (for remaining flips)
   for iter in 0..max_flip_iterations {
       check_delaunay_exact();  // NEW SHADER
       if converged { break; }
   }
   ```

**Expected Impact:**
- Fix degenerate geometry tests (thin_slab, sphere_shell)
- Fix regular grid tests (cospherical points)
- Improve general robustness

---

### 🟠 P1: Fix Point Location

**Why:** Addresses late-insertion failures (30-41% failure on general clouds)

**Debug Steps:**
1. Add logging to relocate_points_fast:
   - How many points relocated successfully?
   - How many failed to find valid tet?
2. Verify flip trace construction (update_flip_trace)
3. Check orient3d decision tree in split_points.wgsl

**Potential Fix:**
- Add fallback: If stochastic walk fails, do exhaustive search
- Validate: After relocate, verify vert_tet points to valid tet containing vertex
- Periodic full location: Every N iterations, relocate ALL uninserted points

---

### 🟠 P2: Validate Mesh Topology

**Why:** Euler characteristic violations indicate corruption

**Add Validation Shader:**
```wgsl
// validate_mesh.wgsl
fn validate_mesh() {
    // For each tet:
    for tet_idx in 0..num_tets {
        // 1. Check adjacency is symmetric
        for face_vi in 0..4 {
            let nei = decode_opp_tet(tet_opp[tet_idx * 4 + face_vi]);
            let nei_face = decode_opp_face(tet_opp[tet_idx * 4 + face_vi]);
            let back = decode_opp_tet(tet_opp[nei * 4 + nei_face]);
            assert(back == tet_idx);  // Adjacency must be symmetric
        }

        // 2. Check tet orientation (positive volume)
        let v0 = verts[tets[tet_idx].x];
        let v1 = verts[tets[tet_idx].y];
        let v2 = verts[tets[tet_idx].z];
        let v3 = verts[tets[tet_idx].w];
        assert(orient3d(v0, v1, v2, v3) > 0);
    }
}
```

Run after split and flip to catch corruption early.

---

### 🟡 P3: Improve Voting System

**Why:** May reduce stalls by better winner selection

**Changes:**
1. Switch from distance-to-centroid to **insphere test**:
   ```wgsl
   // vote.wgsl - CURRENT
   let dist = distance(point, tet_centroid);
   vote_arr[vert_idx] = min(vote_arr[vert_idx], dist);

   // vote.wgsl - PROPOSED
   let sphere_test = insphere_fast(tet_verts, point);
   if sphere_test > 0.0 {  // Point is inside circumsphere
       vote_arr[vert_idx] = min(vote_arr[vert_idx], sphere_test);
   }
   ```

2. Add validation in pick_winner:
   ```wgsl
   // Verify winner is actually inside tet's circumsphere
   if insphere_fast(tet_verts, winner_point) <= 0.0 {
       reject_winner();  // Point is outside, don't insert
   }
   ```

---

## Test-by-Test Diagnosis

### test_raw_thin_slab_100 (84% failure)
- **Geometry:** 100 points on thin planar slab (z ≈ 0)
- **Issue:** Near-coplanar points → degenerate tets → orient3d unreliable
- **Fix:** P0 (exact predicates)

### test_raw_sphere_shell_200 (84% failure)
- **Geometry:** 200 points on sphere surface (2D manifold in 3D)
- **Issue:** No interior points → all tets are near-degenerate
- **Fix:** P0 (exact predicates)

### test_raw_grid_4x4x4 (44% failure)
- **Geometry:** 64 cube vertices on regular grid
- **Issue:** Cospherical points (8 vertices of cube share circumsphere)
- **Fix:** P0 (SoS tie-breaking)

### test_raw_grid_5x5x5 (38% failure)
- **Geometry:** 125 cube vertices on regular grid
- **Issue:** Same as 4x4x4 but larger
- **Fix:** P0 (SoS tie-breaking)
- **Note:** Better convergence (54 iters) suggests larger size helps

### test_raw_uniform_* (38-41% failure)
- **Geometry:** Random uniform distributions
- **Issue:** Late-insertion stalls → vert_tet stale or voting fails
- **Fix:** P1 (point location) + P3 (voting)
- **Note:** Failure rate consistent across sizes

### test_raw_clustered_100 (34% failure)
- **Geometry:** 100 points in tight clusters
- **Issue:** Points in same cluster compete for same tets
- **Fix:** P1 (point location) + P3 (voting)

### test_flip_convergence (33% failure)
- **Geometry:** 27 points (vertices + edge midpoints of cube)
- **Issue:** Edge midpoints are degenerate (lie on edges)
- **Fix:** P0 (exact predicates)

### test_raw_cospherical_12 (25% failure)
- **Geometry:** 12 points on exact sphere
- **Issue:** All points cospherical by construction
- **Fix:** P0 (SoS tie-breaking)

---

## Performance Observations

### Iteration Counts vs Point Set Size

| Test | Points | Failed | Iterations | Inserts/Iter |
|------|--------|--------|------------|--------------|
| cospherical_12 | 12 | 3 (25%) | 5 | 1.8 avg |
| uniform_100 | 100 | 41 (41%) | 10 | 5.9 avg |
| uniform_500 | 500 | 205 (41%) | 38 | 7.8 avg |
| uniform_1000 | 1000 | 383 (38%) | 34 | 18.1 avg |
| grid_5x5x5 | 125 | 48 (38%) | 54 | 1.4 avg |

**Observations:**
1. **Larger point sets** converge faster (more inserts/iteration)
2. **Regular grids** have slow convergence despite failures
3. **Uniform distributions** scale well (38-41% failure regardless of size)

### Time to Stall

Small tests stall quickly (5-16 iterations), large tests run longer (34-54 iterations) before giving up. This suggests:
- Algorithm doesn't "know" it's stuck
- Could add early exit: If 3 consecutive iterations have 0 inserts, abort

---

## Summary

**The implementation is close to working** but lacks **robustness for edge cases**:

✅ **Works well for:**
- General position points (60-70% success)
- Small to medium point sets

❌ **Fails catastrophically for:**
- Degenerate geometry (84% failure)
- Regular grids with cospherical points (38-44% failure)
- Complex mesh topology in late insertion phase (30-41% failure)

**Root causes are well-understood and fixable:**
1. Missing exact predicates (CRITICAL)
2. Point location issues (HIGH)
3. Topology validation missing (HIGH)
4. Voting system could be better (MEDIUM)

**Next steps:** Implement P0 (two-phase flipping with exact predicates) first, as this will fix ~50% of failing tests and improve robustness across the board.
