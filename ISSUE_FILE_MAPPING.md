# Issue to File Mapping: WGSL ↔ CUDA

This document maps each diagnosed issue to the specific source files that need to be examined or modified.

---

## P0: Two-Phase Flipping with Exact Predicates

### Issue: Missing Exact Sphere Test and SoS

**Impact:** 84% failure on degenerate geometry, 38-44% on grids

#### CUDA Implementation

**File:** `gDel3D/GDelFlipping/src/gDel3D/GPU/KerDivision.cu`

**Two-Phase Flipping:**
```cpp
// Lines 721-749: splitAndFlip() function
void GpuDel::splitAndFlip(PredicateBounds predBounds) {
    // ...

    // PHASE 1: Fast floating-point predicates
    doFlippingLoop(SphereFastOrientFast);

    // Mark degenerate cases
    markSpecialTets();

    // PHASE 2: Exact arithmetic + SoS tie-breaking
    doFlippingLoop(SphereExactOrientSoS);
}
```

**Fast Flip Check:**
- **File:** `gDel3D/GDelFlipping/src/gDel3D/GPU/KerPredicates.cu`
- **Function:** `kerCheckDelaunayFast` (lines 398-458)
- **Tests:** Fast floating-point insphere test
- **Flags:** Sets `FLIP_VOTE_VAL` if sphere test fails

**Exact Flip Check:**
- **File:** `gDel3D/GDelFlipping/src/gDel3D/GPU/KerPredicates.cu`
- **Function:** `kerCheckDelaunayExact_Fast` (lines 460-562)
- **Tests:** Exact insphere with adaptive arithmetic
- **SoS:** Uses `doSoSCheck()` for tie-breaking (line 542)

**Mark Special Tets:**
- **File:** `gDel3D/GDelFlipping/src/gDel3D/GPU/KerDivision.cu`
- **Function:** `kerMarkSpecialTets` (lines 653-680)
- **Purpose:** Clears special adjacency markers before exact phase

**Exact Predicates Library:**
- **File:** `gDel3D/GDelFlipping/src/gDel3D/Geometry/predicates.c`
- **Author:** Jonathan Shewchuk
- **Functions:**
  - `orient3dexact()` - Exact orientation test
  - `insphereexact()` - Exact in-sphere test
  - `orient3dadapt()` - Adaptive orientation (fast → exact)
  - `inspherefast()` - Fast floating-point sphere test

**SoS Implementation:**
- **File:** `gDel3D/GDelFlipping/src/gDel3D/GPU/KerPredicates.cu`
- **Function:** `doSoSCheck()` (lines 78-115)
- **Method:** Symbolic perturbation using vertex indices
- **Reference:** Edelsbrunner & Mücke (1990)

#### WGPU Current Implementation

**Fast Phase (EXISTS):**
- **File:** `src/shaders/check_delaunay_fast.wgsl`
- **Function:** `check_delaunay_fast()` (lines 39-135)
- **Tests:** Fast floating-point insphere approximation
- **Issues:**
  - Line 89-95: Approximates circumsphere with centroid + radius
  - Line 121: Uses simple distance comparison (no exact fallback)

**Exact Phase (MISSING):**
- **File:** `src/shaders/check_delaunay_exact.wgsl` ❌ **DOES NOT EXIST**
- **Needed:** Port of `kerCheckDelaunayExact_Fast`

**Mark Special (EXISTS but NOT WIRED):**
- **File:** `src/shaders/mark_special_tets.wgsl` ✅ EXISTS
- **Port of:** `kerMarkSpecialTets` from KerDivision.cu:653-680
- **Status:** Pipeline created but not called in phase1.rs
- **Location in phase1.rs:** Line 154-157 (commented out, needs two-phase loop first)

#### Required Changes

**1. Create check_delaunay_exact.wgsl**

Port from `KerPredicates.cu:460-562`:
```wgsl
// NEW FILE: src/shaders/check_delaunay_exact.wgsl

// Exact insphere test using Shewchuk's adaptive arithmetic
fn insphere_exact(pa: vec3<f32>, pb: vec3<f32>, pc: vec3<f32>,
                  pd: vec3<f32>, pe: vec3<f32>) -> f32 {
    // Adaptive filter:
    // 1. Try fast floating-point
    // 2. If uncertain, use exact arithmetic
    // 3. If still zero, use SoS
}

// SoS tie-breaking (port from doSoSCheck, lines 78-115)
fn sos_insphere(tet_vi: array<u32, 4>, point_vi: u32) -> i32 {
    // Symbolic perturbation based on vertex indices
    // Returns deterministic sign for degenerate cases
}

@compute @workgroup_size(64)
fn check_delaunay_exact(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Port of kerCheckDelaunayExact_Fast
    // Same structure as check_delaunay_fast.wgsl but with exact tests
}
```

**Reference CUDA lines:**
- `KerPredicates.cu:460-562` - Main kernel structure
- `KerPredicates.cu:78-115` - SoS implementation
- `predicates.c:2299-2448` - insphereexact() implementation
- `predicates.c:1649-1814` - orient3dexact() implementation

**2. Modify phase1.rs**

Current (line 153-234):
```rust
// Single loop with fast checks only
for flip_iter in 0..config.max_flip_iterations {
    dispatch_check_delaunay_fast();
    // ...
}
```

Required:
```rust
// PHASE 1: Fast checks
let mut fast_converged = false;
for flip_iter in 0..config.max_fast_flip_iterations {
    dispatch_check_delaunay_fast();
    // ... (existing flip logic)

    if flip_count == 0 {
        fast_converged = true;
        break;
    }
}

// Mark special tets (degenerate cases flagged during fast phase)
dispatch_mark_special_tets();

// PHASE 2: Exact checks for remaining violations
if !fast_converged {
    for flip_iter in 0..config.max_exact_flip_iterations {
        dispatch_check_delaunay_exact();  // NEW DISPATCH
        // ... (same flip logic)

        if flip_count == 0 { break; }
    }
}
```

**Reference CUDA:** `GpuDelaunay.cu:721-749` (splitAndFlip)

**3. Add dispatch_check_delaunay_exact**

- **File:** `src/gpu/dispatch.rs`
- **Add method:** `dispatch_check_delaunay_exact()` (similar to line 235-254)
- **Pipeline:** `src/gpu/pipelines.rs` (add exact pipeline)

---

## P1: Point Location Failures

### Issue: vert_tet becomes stale, late-insertion failures

**Impact:** 30-41% failure on general point clouds

#### CUDA Implementation

**Point Location (Stochastic Walk):**
- **File:** `gDel3D/GDelFlipping/src/gDel3D/GPU/KerPredicates.cu`
- **Function:** `kerPointLocationFast` (lines 117-208)
- **Method:**
  - Stochastic walk using orient3d tests (lines 157-196)
  - Maximum 200 steps (line 134)
  - Falls back to current tet if walk fails (line 198)

**Relocation After Flipping:**
- **File:** `gDel3D/GDelFlipping/src/gDel3D/GPU/KerPredicates.cu`
- **Function:** `kerRelocatePointsFast` (lines 305-396)
- **Method:**
  - Uses flip trace to follow tet replacements (lines 338-372)
  - Updates vert_tet for points in flipped region (lines 374-389)
  - Called AFTER all flipping is done (line 742 in GpuDelaunay.cu)

**Flip Trace Construction:**
- **File:** `gDel3D/GDelFlipping/src/gDel3D/GPU/KerDivision.cu`
- **Function:** `kerUpdateFlipTrace` (lines 562-612)
- **Purpose:** Maps old tet indices → flip chain heads
- **Structure:** Processes flip batches in REVERSE order (line 575)

#### WGPU Current Implementation

**Point Location:**
- **File:** `src/shaders/point_location.wgsl`
- **Function:** `point_location()` (lines 62-134)
- **Port of:** `kerPointLocationFast` (KerPredicates.cu:117-208)
- **Issues:**
  - Line 94: TetOpp decoding (verified correct: `>> 5u`)
  - Line 103-131: Walk logic matches CUDA
  - May need better fallback if walk fails

**Relocation:**
- **File:** `src/shaders/relocate_points_fast.wgsl`
- **Function:** `relocate_points_fast()` (lines 34-89)
- **Port of:** `kerRelocatePointsFast` (KerPredicates.cu:305-396)
- **Issues:**
  - Lines 51-75: Flip trace following logic
  - May have bugs in chain traversal

**Flip Trace Update:**
- **File:** `src/shaders/update_flip_trace.wgsl`
- **Function:** `update_flip_trace()` (lines 27-52)
- **Port of:** `kerUpdateFlipTrace` (KerDivision.cu:562-612)
- **Called from:** `src/phase1.rs:249-254` (reverse iteration)

**Split Points (Updates vert_tet before split):**
- **File:** `src/shaders/split_points.wgsl`
- **Function:** `split_points()` (lines 97-166)
- **Port of:** `kerSplitPointsFast` (KerPredicates.cu:210-283)
- **Purpose:** Updates vert_tet for vertices whose tets are about to split
- **Method:** Orient3d decision tree using SplitFaces/SplitNext lookup tables

#### Required Changes

**1. Add Diagnostic Logging to relocate_points_fast.wgsl**

Add counters to track:
```wgsl
// Line ~40, add to counters buffer
var<storage, read_write> debug_counters: array<atomic<u32>>;

// After line 75 (successful relocation)
atomicAdd(&debug_counters[0], 1u);  // Count successful relocations

// Add at line ~85 (if walk fails)
atomicAdd(&debug_counters[1], 1u);  // Count failed relocations
```

**2. Verify Flip Trace Logic**

Compare WGSL (update_flip_trace.wgsl:27-52) against CUDA (KerDivision.cu:562-612):
- Check reverse iteration matches
- Verify tet_to_flip indexing
- Validate chain construction

**3. Add Fallback Point Location**

In `src/phase1.rs` after relocate:
```rust
// After line 261 (relocate completes)
let counters = state.buffers.read_counters(device, queue).await;
let relocated = counters.debug[0];
let failed = counters.debug[1];

if failed > num_uninserted / 10 {  // >10% failed
    // Fallback: Full point location for all uninserted
    let mut encoder = device.create_command_encoder(&Default::default());
    state.dispatch_locate(&mut encoder, queue, num_uninserted);
    queue.submit(Some(encoder.finish()));
}
```

**4. Validate vert_tet After Update**

Add validation shader:
```wgsl
// NEW: validate_vert_tet.wgsl
@compute
fn validate_vert_tet() {
    let vert_idx = global_id.x;
    let tet_idx = vert_tet[vert_idx];
    let point = points[uninserted[vert_idx]];

    // Check if point is actually in tet
    let tet = tets[tet_idx];
    let o0 = orient3d(point, verts[tet.y], verts[tet.z], verts[tet.w]);
    let o1 = orient3d(verts[tet.x], point, verts[tet.z], verts[tet.w]);
    let o2 = orient3d(verts[tet.x], verts[tet.y], point, verts[tet.w]);
    let o3 = orient3d(verts[tet.x], verts[tet.y], verts[tet.z], point);

    if (o0 < 0.0 || o1 < 0.0 || o2 < 0.0 || o3 < 0.0) {
        atomicAdd(&invalid_vert_tet_count, 1u);
    }
}
```

---

## P2: Mesh Topology Validation

### Issue: Euler characteristic violations indicate corruption

**Impact:** Negative Euler values (-19, -44) in thin_slab and clustered tests

#### CUDA Implementation

**No Explicit Validation in CUDA**

CUDA code assumes:
- Split operations maintain adjacency invariants
- Flip operations preserve topology
- No runtime validation (relies on correctness of implementation)

**However, CUDA has defensive checks:**
- **File:** `gDel3D/GDelFlipping/src/gDel3D/GPU/KerDivision.cu`
- **Split concurrent detection:** Lines 140-162
  - Checks if neighbor was also split (via tetToVert)
  - Uses free_arr to find new tet if concurrent split detected

#### WGPU Current Implementation

**Split Operation:**
- **File:** `src/shaders/split.wgsl`
- **Function:** `split()` (lines 122-247)
- **Port of:** `kerSplitTetra` (KerDivision.cu:118-225)
- **Critical sections:**
  - Lines 158-168: Vertex ordering (VERIFIED CORRECT as of 2026-03-03)
  - Lines 190-212: Internal adjacency (uses IntSplitFaceOpp table)
  - Lines 214-247: External adjacency update
    - Line 216: TODO mentions race condition
    - Lines 217-231: External neighbor updates (may have bugs)

**Flip Operation:**
- **File:** `src/shaders/flip.wgsl`
- **Function:** `flip()` (lines 82-186)
- **Port of:** `kerFlip` (KerDivision.cu:339-458)
- **Critical sections:**
  - Lines 82-115: Read old tet configuration
  - Lines 117-155: Create 3 new tets (2-3 flip)
  - Lines 157-180: Set adjacency for new tets
  - Issues: External neighbors not updated in flip.wgsl itself

**Update Opp (Fix External Adjacency After Flip):**
- **File:** `src/shaders/update_opp.wgsl`
- **Function:** `update_opp()` (lines 41-106)
- **Port of:** `kerUpdateOpp` (KerDivision.cu:460-522)
- **Purpose:** Updates external neighbors to point back to new flipped tets
- **Called from:** `src/phase1.rs:223-225`

#### Potential Bug Locations

**1. Split External Adjacency (split.wgsl:217-231)**

Current code is commented/disabled:
```wgsl
// Lines 217-231: External adjacency update
// TODO comment at line 216: "Fix voting system to ensure only 1 winner per tet"
// External neighbor updates are DISABLED or incomplete
```

**CUDA Reference:** `KerDivision.cu:140-162`
```cpp
// Check for concurrent split
int splitVertIdx = tetToVert[ neiTetIdx ];

if ( -1 != splitVertIdx ) {  // Neighbor was also split
    // Use free_arr to find the new tet
    const int neiTetIdx = freeArr[ splitVertIdx ];
    // Update to point to new split tet instead
}
```

**Required:** Verify WGSL implements concurrent split detection

**2. Flip Adjacency (flip.wgsl:157-180)**

Check if internal adjacency is correct:
```wgsl
// Lines 157-180: Set opp for 3 new tets
// Compare against CUDA KerDivision.cu:389-419
```

**3. Update Opp After Flip (update_opp.wgsl:41-106)**

Verify message passing logic:
```wgsl
// Lines 54-73: Decode flip message
// Lines 75-106: Update neighbor adjacency
// Compare against CUDA KerDivision.cu:460-522
```

#### Required Changes

**1. Create validate_mesh.wgsl**

```wgsl
// NEW FILE: src/shaders/validate_mesh.wgsl

@group(0) @binding(0) var<storage, read> tets: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> tet_opp: array<u32>;
@group(0) @binding(2) var<storage, read> tet_info: array<u32>;
@group(0) @binding(3) var<storage, read_write> error_count: array<atomic<u32>>;

fn decode_opp_tet(packed: u32) -> u32 { return packed >> 5u; }
fn decode_opp_face(packed: u32) -> u32 { return packed & 3u; }

@compute @workgroup_size(64)
fn validate_mesh(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tet_idx = global_id.x;
    if tet_idx >= arrayLength(&tets) { return; }

    // Skip dead tets
    if tet_info[tet_idx] == 0u { return; }

    // Check adjacency symmetry for each face
    for (var face_vi = 0u; face_vi < 4u; face_vi++) {
        let packed = tet_opp[tet_idx * 4u + face_vi];
        let nei_tet = decode_opp_tet(packed);
        let nei_face = decode_opp_face(packed);

        // Check neighbor points back to us
        let back_packed = tet_opp[nei_tet * 4u + nei_face];
        let back_tet = decode_opp_tet(back_packed);
        let back_face = decode_opp_face(back_packed);

        if back_tet != tet_idx || back_face != face_vi {
            atomicAdd(&error_count[0], 1u);  // Asymmetric adjacency
        }
    }

    // TODO: Add orientation check (positive volume)
    // TODO: Add duplicate tet detection
}
```

**2. Call Validation After Split/Flip**

In `src/phase1.rs`:
```rust
// After line 142 (split completes)
#[cfg(debug_assertions)]
{
    let mut encoder = device.create_command_encoder(&Default::default());
    state.dispatch_validate_mesh(&mut encoder, queue);
    queue.submit(Some(encoder.finish()));

    let errors = state.buffers.read_error_count(device, queue).await;
    if errors[0] > 0 {
        panic!("Split created {} adjacency errors!", errors[0]);
    }
}

// After line 225 (update_opp completes)
#[cfg(debug_assertions)]
{
    // Validate again after flipping
    // ...
}
```

**3. Fix Split External Adjacency**

In `src/shaders/split.wgsl`, lines 217-231, implement concurrent split detection:

```wgsl
// Port from CUDA KerDivision.cu:140-162
for (var i = 0u; i < 4u; i++) {
    let ext_opp = ext_opps[i];
    let nei_tet = decode_opp_tet(ext_opp);
    let nei_face = decode_opp_face(ext_opp);

    // Check if neighbor was also split (concurrent split detection)
    let nei_split_idx = tet_to_vert[nei_tet];

    var actual_nei_tet = nei_tet;
    if nei_split_idx != 0xFFFFFFFFu {  // Neighbor was split
        // Find the new tet from split operation
        // Use free_arr lookup (port from CUDA line 158)
        actual_nei_tet = /* lookup in free_arr */;
    }

    // Update neighbor to point back to us
    set_opp_at(actual_nei_tet, nei_face, encode_opp(new_tets[i], i));
}
```

**Reference CUDA:** `KerDivision.cu:140-162`

---

## P3: Voting System Limitations

### Issue: Uses distance-to-centroid instead of insphere test

**Impact:** May pick points outside circumsphere, leading to failed insertions

#### CUDA Implementation

**Voting:**
- **File:** `gDel3D/GDelFlipping/src/gDel3D/GPU/KerDivision.cu`
- **Function:** `kerVoteForPoint` (lines 39-90)
- **Method:** (lines 65-78)
  - Computes **circumsphere** of tet (not just centroid)
  - Tests if point is **inside circumsphere** using insphere test
  - Votes based on sphere test result (not distance)

```cpp
// Line 65-78 in kerVoteForPoint
Real sphere = inspherefast(
    tetVert[0]._p, tetVert[1]._p,
    tetVert[2]._p, tetVert[3]._p,
    vertVec[ vIdx ]._p
);

if (sphere > 0.0) {  // Point is inside circumsphere
    // Vote for this point
    atomicMin(&vertSphereArr[vIdx], sphere);
}
```

**Pick Winner:**
- **File:** `gDel3D/GDelFlipping/src/gDel3D/GPU/KerDivision.cu`
- **Function:** `kerPickWinnerPoint` (lines 92-116)
- **Method:** **Vertex-parallel** iteration
  - Each vertex (in parallel) checks if it won any tet
  - Compares its vote value against tet's vote value
  - If match, vertex claims the tet

#### WGPU Current Implementation

**Voting:**
- **File:** `src/shaders/vote.wgsl`
- **Function:** `vote()` (lines 40-102)
- **Method:** Lines 66-91
  - Computes tet **centroid** (simple average of vertices)
  - Computes **distance** from point to centroid
  - Uses distance as vote value (smaller = better)

```wgsl
// Lines 66-82: Centroid calculation
let centroid = (v0 + v1 + v2 + v3) / 4.0;

// Lines 84-87: Distance calculation
let point_pos = points[vert_idx];
let dist = distance(point_pos, centroid);

// Lines 89-91: Vote with distance
atomicMin(&vert_sphere[vert_idx], bitcast<i32>(dist));
```

**Pick Winner:**
- **File:** `src/shaders/pick_winner.wgsl`
- **Function:** `pick_winner()` (lines 56-120)
- **Method:** **Tet-parallel** iteration (design difference)
  - Each tet checks all vertices to find best candidate
  - Picks vertex with minimum vote value
  - Different from CUDA's vertex-parallel approach

#### Required Changes

**1. Replace Distance with Insphere Test in vote.wgsl**

Current (lines 66-91):
```wgsl
// CURRENT - WRONG
let centroid = (v0 + v1 + v2 + v3) / 4.0;
let dist = distance(point_pos, centroid);
atomicMin(&vert_sphere[vert_idx], bitcast<i32>(dist));
```

Replace with:
```wgsl
// NEW - Port from CUDA kerVoteForPoint:65-78
let sphere_test = insphere_fast(v0, v1, v2, v3, point_pos);

if sphere_test > 0.0 {  // Point is inside circumsphere
    // Smaller sphere_test = closer to sphere surface = better vote
    atomicMin(&vert_sphere[vert_idx], bitcast<i32>(sphere_test));
}
```

**2. Add insphere_fast() to vote.wgsl**

Port from `predicates.c:2299-2448` (fast version):
```wgsl
// Add near top of vote.wgsl
fn insphere_fast(
    pa: vec3<f32>, pb: vec3<f32>, pc: vec3<f32>,
    pd: vec3<f32>, pe: vec3<f32>
) -> f32 {
    // Fast floating-point in-sphere test
    // Returns:
    //   > 0 if pe is inside circumsphere of (pa, pb, pc, pd)
    //   < 0 if pe is outside
    //   ≈ 0 if pe is on sphere (degenerate)

    let aex = pa.x - pe.x;
    let bex = pb.x - pe.x;
    let cex = pc.x - pe.x;
    let dex = pd.x - pe.x;
    let aey = pa.y - pe.y;
    let bey = pb.y - pe.y;
    let cey = pc.y - pe.y;
    let dey = pd.y - pe.y;
    let aez = pa.z - pe.z;
    let bez = pb.z - pe.z;
    let cez = pc.z - pe.z;
    let dez = pd.z - pe.z;

    let ab = aex * bey - bex * aey;
    let bc = bex * cey - cex * bey;
    let cd = cex * dey - dex * cey;
    let da = dex * aey - aex * dey;
    let ac = aex * cey - cex * aey;
    let bd = bex * dey - dex * bey;

    let abc = aez * bc - bez * ac + cez * ab;
    let bcd = bez * cd - cez * bd + dez * bc;
    let cda = cez * da + dez * ac + aez * cd;
    let dab = dez * ab + aez * bd + bez * da;

    let alift = aex * aex + aey * aey + aez * aez;
    let blift = bex * bex + bey * bey + bez * bez;
    let clift = cex * cex + cey * cey + cez * cez;
    let dlift = dex * dex + dey * dey + dez * dez;

    return (dlift * abc - clift * dab) + (blift * cda - alift * bcd);
}
```

**Reference:** Shewchuk's `predicates.c:2299-2448`

**3. Consider Switching to Vertex-Parallel pick_winner**

Current WGSL uses tet-parallel (design difference from CUDA).

To match CUDA exactly, create new `pick_winner_vertex_parallel.wgsl`:
```wgsl
// Port of kerPickWinnerPoint (KerDivision.cu:92-116)
@compute @workgroup_size(64)
fn pick_winner_vertex_parallel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vert_idx = global_id.x;

    // Check if this vertex won any tet
    let my_vote = vert_sphere[vert_idx];

    // Scan all tets to find which one picked this vertex
    for (var tet_idx = 0u; tet_idx < num_tets; tet_idx++) {
        let tet_vote = tet_sphere[tet_idx];

        if tet_vote == my_vote {
            // This tet picked me! Claim it.
            // (Need to handle ties)
        }
    }
}
```

**Note:** This may be expensive (each vertex scans all tets). Current tet-parallel approach may actually be better for GPU.

---

## Additional File References

### Lookup Tables

**CUDA:**
- **File:** `gDel3D/GDelFlipping/src/gDel3D/GPU/GPUDecl.h`
- **Tables:**
  - `TetViAsSeenFrom[4][3]` (lines 40-45) - Vertex permutations for split
  - `IntSplitFaceOpp[4][6]` (lines 47-52) - Internal adjacency for split
  - `SplitFaces[11][3]` (lines 77-89) - Orient3d decision tree for split_points
  - `SplitNext[11][2]` (lines 91-103) - Navigation for split_points tree

**WGSL:**
- **File:** `src/shaders/constants.wgsl`
- **Tables:** All lookup tables ported ✅
- **Verified:** 2026-03-02 agent audit confirmed correctness

### Constants

**CUDA:**
- **File:** `gDel3D/GDelFlipping/src/gDel3D/GPU/KerCommon.h`
- **Constants:**
  - `MEAN_VERTEX_DEGREE = 8` (line 56)
  - `OPP_SPECIAL = 8` (bit flag for special adjacency)
  - `OPP_SPHERE_FAIL = 16` (bit flag for sphere test failure)

**WGSL:**
- **File:** `src/shaders/constants.wgsl`
- **Constants:** All match CUDA ✅
- **Verified:** 2026-03-02 agent audit fixed MEAN_VERTEX_DEGREE

### Buffer Layouts

**CUDA:**
- **File:** `gDel3D/GDelFlipping/src/gDel3D/CommonTypes.h`
- **TetOpp Encoding:** Lines 240-351
  - `makeOppVal()` - line 248: `(tet_idx << 5) | vi`
  - `getOppValTet()` - line 265: `val >> 5`
  - `getOppValVi()` - line 264: `val & 3`

**WGSL:**
- **Multiple files** use TetOpp encoding
- **Verified correct:** 2026-03-03 (5-bit shift)
- **Exception fixed:** split.wgsl now uses 5-bit (was 2-bit)

---

## Summary Table

| Issue | Priority | WGSL Files | CUDA Files | Status |
|-------|----------|------------|------------|--------|
| **Exact predicates** | P0 | check_delaunay_exact.wgsl ❌ | KerPredicates.cu:460-562, predicates.c | Missing |
| **SoS tie-breaking** | P0 | (in check_delaunay_exact.wgsl) ❌ | KerPredicates.cu:78-115 | Missing |
| **Mark special tets** | P0 | mark_special_tets.wgsl ✅ | KerDivision.cu:653-680 | Exists, not wired |
| **Two-phase flipping** | P0 | phase1.rs | GpuDelaunay.cu:721-749 | Need refactor |
| **Point location** | P1 | point_location.wgsl ✅ | KerPredicates.cu:117-208 | Verify correctness |
| **Relocate points** | P1 | relocate_points_fast.wgsl ✅ | KerPredicates.cu:305-396 | Needs debugging |
| **Flip trace** | P1 | update_flip_trace.wgsl ✅ | KerDivision.cu:562-612 | Verify correctness |
| **Split points** | P1 | split_points.wgsl ✅ | KerPredicates.cu:210-283 | Verify correctness |
| **Mesh validation** | P2 | validate_mesh.wgsl ❌ | (not in CUDA) | Need to create |
| **Split adjacency** | P2 | split.wgsl:217-231 ⚠️ | KerDivision.cu:140-162 | Incomplete/disabled |
| **Insphere voting** | P3 | vote.wgsl:66-91 ⚠️ | KerDivision.cu:65-78 | Wrong metric |
| **Vertex-parallel pick** | P3 | pick_winner.wgsl ⚠️ | KerDivision.cu:92-116 | Design difference |

**Legend:**
- ✅ Exists and believed correct
- ⚠️ Exists but has issues
- ❌ Missing/needs creation
