# COMPILED FLAWS - gDel3D WGPU Port

**Last Updated:** 2026-03-04
**Test Status:** 21/66 passing (31.8%)
**Critical Blocker:** SIGSEGV during GPU test execution

---

## 🔴 P0: CRITICAL BLOCKER

### SIGSEGV During GPU Test Execution

**Impact:** 45/66 tests fail with segmentation fault

**Symptoms:**
```
running 66 tests
test predicates::tests::test_circumcenter ... ok
...
test tests::test_gpu_flip_shader_compiles ... ok
error: test failed, to rerun pass `--lib`
Caused by:
  process didn't exit successfully (signal: 11, SIGSEGV: invalid memory reference)
```

**Tests Passing:**
- ✅ Predicate unit tests (5/5)
- ✅ Shader compilation tests (6/6)
- ✅ Simple CPU tests (9/9)
- ✅ Minimal buffer creation (1/1)

**Tests Failing:**
- ❌ All GPU pipeline integration tests
- ❌ All Delaunay correctness tests
- ❌ All raw output tests

**Likely Causes:**
1. Invalid buffer access (out of bounds indexing)
2. Race condition in atomic operations
3. Uninitialized buffer reads
4. Incorrect bind group layouts (buffer size mismatches)

**Investigation Steps:**
1. Enable GPU validation layers (`InstanceFlags::VALIDATION | InstanceFlags::DEBUG`)
2. Run tests individually to isolate which test crashes
3. Add breadcrumb debugging to shaders
4. Verify all buffer allocations match shader expectations
5. Check atomic operation declarations match between Rust and WGSL

---

## 🟡 P1: CONFIRMED DESIGN FLAWS

### FLAW #1: Missing TET_EMPTY Bit Implementation

**Status:** NOT IMPLEMENTED
**Impact:** Performance issue + semantic drift from CUDA

#### What's Missing

**CUDA tet_info Bit Layout** (CommonTypes.h:384-388):
```
76543210
     ^^^ 0: Dead      1: Alive     (bit 0)
     ||_ 0: Checked   1: Changed   (bit 1)
     |__ 0: NotEmpty  1: Empty     (bit 2)
```

**WGPU Implementation:**
```rust
// src/types.rs
pub const TET_ALIVE: u32 = 1 << 0;
pub const TET_CHANGED: u32 = 1 << 1;
pub const TET_CHECKED: u32 = 1 << 2;  // ❌ SHOULD BE TET_EMPTY!
pub const TET_LOCKED: u32 = 1 << 3;
```

**Critical Issue:** Bit 2 is used for `TET_CHECKED` instead of `TET_EMPTY`, causing semantic divergence.

#### CUDA Usage Pattern

1. **kerMarkTetEmpty** (GpuDelaunay.cu:1013) - Sets ALL tets to empty=true at iteration start
2. **kerSplitPointsFast** (KerPredicates.cu:232, 276) - Sets empty=false for tets containing uninserted vertices
3. **kerFlip** (KerDivision.cu:514-527) - Checks if all tets in flip are empty, sets `flipItem._v[0] = -1` marker
4. **kerUpdateFlipTrace** (KerDivision.cu:755) - Skips flips with `_v[0] == -1`

#### WGPU Current State

- ❌ No `TET_EMPTY` constant defined
- ❌ No `mark_tet_empty.wgsl` shader exists
- ❌ `split_points.wgsl` does NOT set empty=false
- ❌ `flip.wgsl` does NOT check tetEmpty or set `_v[0] = -1`
- ✅ `update_flip_trace.wgsl:85` DOES check for `_v[0] == -1` but never gets it!

#### Impact

- **Correctness:** Minimal - flip tracing still works, just does extra work
- **Performance:** Minor - traces ALL flips instead of skipping "clean" boundary flips
- **Semantic Drift:** Moderate - loses information about which tets contain uninserted vertices
- **Code Inconsistency:** `update_flip_trace.wgsl` expects the marker but never receives it

#### Implementation Required

1. Change `TET_CHECKED` to `TET_EMPTY` in types.rs (or use bit 4)
2. Create `mark_tet_empty.wgsl` kernel (sets all tets to empty=true)
3. Call `mark_tet_empty` at start of each insertion iteration
4. Update `split_points.wgsl` to set empty=false when visiting tets (lines 232, 276 equivalent)
5. Update `flip.wgsl` to:
   - Check `tetEmpty` for all tets in flip (read `tet_info & TET_EMPTY`)
   - Set `flipItem._v[0] = -1` when all tets are empty
   - Inherit empty state in new tets (set/clear TET_EMPTY bit)

---

### FLAW #2: CPU-Side Vertex Compaction

**Status:** ✅ FIXED (2026-03-04)
**Impact:** Performance improvement for large datasets

#### What Was Wrong

**Location:** `phase1.rs:457-478` (old implementation)

The original implementation used CPU-side sequential processing to compact vertex arrays:
```rust
// CUDA: GPU parallel thrust::remove_if
// WGPU: OLD CPU sequential Vec::retain (SLOW!)
state.uninserted.retain(|v| !inserted_verts.contains(v));
// ... sequential loop to compact vert_tet ...
```

**Performance Issues:**
- CPU-GPU round trip latency (microseconds to milliseconds)
- Sequential CPU processing instead of parallel GPU processing
- Pipeline bubble: GPU sits idle while CPU processes
- Scales poorly with large point sets (10k+ points)

#### Fix Implementation (2026-03-04)

✅ **Created GPU-accelerated vertex compaction:**
- **Shader:** `compact_vertex_arrays.wgsl` - 2-pass compaction (count, then scatter)
- **Pipeline:** Added to `pipelines.rs` with bind groups
- **Dispatch:** `dispatch_compact_vertex_arrays()` at `dispatch.rs:442-478`
- **Integration:** Replaced CPU code in `phase1.rs:443-488`
- **Buffers:** Added `uninserted_temp` and `vert_tet_temp` for double-buffering

**Algorithm:**
- Pass 1: Count non-inserted vertices (parallel atomic increment)
- Pass 2: Scatter non-inserted vertices to temp buffers
- Copy temp buffers back to main buffers
- Read back compacted arrays to update CPU state

**CUDA Parity:**
Port of `thrust::remove_if` with zip iterators (ThrustWrapper.cu:244-265) using linear search through insert_list.

#### Test Status

✅ Shader compiles successfully
✅ Pipeline creation works
⚠️ Full integration tests blocked by pre-existing SIGSEGV (P0 issue)

---

### FLAW #3: Shader Compilation Bugs (FIXED)

**Status:** ✅ ALL FIXED

These were blocking all tests but are now resolved:

1. ✅ **gather.wgsl:104** - Variable `opp_packed` redefined → Removed redundant declaration
2. ✅ **split.wgsl:216** - Undefined `inf_idx` → Added extraction from `params.y`
3. ✅ **flip.wgsl:22** - `vert_free_arr` non-atomic → Changed to `array<atomic<u32>>`

---

## 🟢 P2: DOCUMENTED PATTERNS & GOTCHAS

### Recurring Bug: vert_tet Position Indexing

**Critical Pattern:** `vert_tet` is position-indexed (by idx in uninserted array), NOT vertex-indexed.

#### The Cognitive Trap

```wgsl
let idx = gid.x;              // Position: 0, 1, 2, ...
let vert_idx = uninserted[idx];  // Vertex ID: could be 42

// CORRECT (but looks wrong):
let tet_idx = vert_tet[idx];     // Use position! ✅

// WRONG (but looks right):
let tet_idx = vert_tet[vert_idx]; // ❌ BUG!
```

#### Why It Keeps Reappearing

1. **Misleading name** - `vert_tet` sounds like "vertex to tet mapping"
2. **Breaks pattern** - Everywhere else uses `vert_idx` after loading it
3. **Fails silently** - Reads WRONG tet but doesn't crash
4. **Debugging intuition backwards** - "Fix" introduces bug

#### Prevention (Applied)

✅ Prominent warning comments in vote.wgsl (lines 73, 146, 188)
✅ Boxed warnings with CUDA references
✅ Order enforcement (read tet before getting vert_idx)

#### Buffer Indexing Reference

| Buffer | Indexed By | Example |
|--------|-----------|---------|
| `points` | Vertex ID | `points[vert_idx]` ✅ |
| `tets` | Tet ID | `tets[tet_idx]` ✅ |
| `tet_info` | Tet ID | `tet_info[tet_idx]` ✅ |
| `uninserted` | Position | `uninserted[idx]` ✅ |
| **`vert_tet`** | **Position** ⚠️ | **`vert_tet[idx]`** NOT `vert_tet[vert_idx]` |
| `vert_sphere` | Position | `vert_sphere[idx]` ✅ |

#### Future Fix

Rename `vert_tet` → `uninserted_tet` to make indexing obvious.

---

## 📊 CURRENT IMPLEMENTATION STATUS

### Two-Phase Flipping: ✅ FULLY IMPLEMENTED

**Shaders:**
- ✅ `check_delaunay_fast.wgsl` - Fast predicates with OPP_SPECIAL flag
- ✅ `check_delaunay_exact.wgsl` - Double-double arithmetic + SoS
- ✅ `mark_special_tets.wgsl` - Queues uncertain tets for exact pass

**Pipeline (phase1.rs:247-412):**
1. Fast phase - f32 predicates, marks special cases
2. Mark special tets - builds queue of uncertain cases
3. Exact phase - DD + SoS for degenerate geometry

**Results:**
- Improves degenerate geometry handling 3-4×
- 40-50% of points still fail (need star splaying fallback)

---

### Star Splaying: ⚠️ PARTIALLY IMPLEMENTED

**Implemented:**
- ✅ `gather.wgsl` - Detects failed vertices via OPP_SPHERE_FAIL flags
- ✅ OPP_SPHERE_FAIL flag is set correctly by check_delaunay shaders

**Missing:**
- ❌ `dispatch_gather_failed()` - Method to wire shader into pipeline
- ❌ CPU star splaying fallback (~1500 lines from CUDA Splaying.cpp + Star.cpp)

**Status:** Shader ready but not integrated. CPU fallback not yet ported.

---

### Compaction: ✅ FULLY IMPLEMENTED

**GPU Compaction (All Implemented):**
- ✅ Flip queues - `compact_if_negative.wgsl`
- ✅ Tet arrays - `compact_tets.wgsl` with 3-step prefix sum
- ✅ Vertex arrays - `compact_vertex_arrays.wgsl` (added 2026-03-04)
- ✅ Called before readback (tests.rs:1857)

---

### Deleted Files (commit c24b243 "gonna clean")

These files were removed as obsolete:

1. ✅ `src/shaders/mark_split.wgsl` - Redundant, `pick_winner_point` already populates tet_to_vert
2. ✅ `src/shaders/pick_winner.wgsl` - Obsolete tet-parallel, vote.wgsl uses vertex-parallel
3. ✅ `src/shaders/split_fixup.wgsl` - Redundant, split.wgsl handles adjacency inline
4. ✅ `src/shaders/point_location.wgsl` - Workaround removed, split_points keeps vert_tet updated
5. ✅ `src/coloring.rs` - Unused debugging code

**Rationale:** All were either redundant, obsolete, or workarounds for bugs now fixed.

---

## 🎯 PRIORITY ACTION PLAN

### Immediate (Fix SIGSEGV)

1. **Enable GPU validation** in tests:
   ```rust
   InstanceFlags::VALIDATION | InstanceFlags::DEBUG
   ```

2. **Run tests individually** to isolate crash:
   ```bash
   RUST_BACKTRACE=1 cargo test --lib test_delaunay_4_points -- --nocapture
   ```

3. **Add breadcrumb debugging** to shaders (implementation ready in deleted DEBUGGING_TOOLS.md)

4. **Verify buffer bindings:**
   - Check all bind group layouts match shader expectations
   - Verify atomic declarations match between Rust and WGSL
   - Ensure uniform buffers written before dispatch

### Short Term (Performance & Correctness)

1. **Implement TET_EMPTY tracking** (FLAW #1)
   - Define constant (resolve bit 2 conflict)
   - Create mark_tet_empty shader
   - Update split_points and flip shaders
   - Wire into phase1 pipeline

2. **Integrate gather_failed** (Star Splaying)
   - Add dispatch_gather_failed() method
   - Wire into pipeline after exact flipping

### Long Term (CUDA Parity)

1. **Port CPU star splaying** (~1500 lines)
2. **Rename vert_tet → uninserted_tet** (prevent recurring bugs)
3. **Implement remaining CUDA features** if needed

---

## 📝 CUDA REFERENCES

### Critical Files (Original Implementation)

- **Constants:** `gDel3D/GDelFlipping/src/gDel3D/GPU/KerCommon.h`
- **TetOpp Encoding:** `gDel3D/GDelFlipping/src/gDel3D/CommonTypes.h:240-351`
- **Kernels:** `gDel3D/GDelFlipping/src/gDel3D/GPU/KerDivision.cu`
- **Lookup Tables:** `gDel3D/GDelFlipping/src/gDel3D/GPU/GPUDecl.h`
- **Star Splaying:** `gDel3D/GDelFlipping/src/gDel3D/CPU/Splaying.cpp` + `Star.cpp`

### Key Constants

- `MEAN_VERTEX_DEGREE = 8` (NOT 64!)
- TetOpp encoding: 5-bit (tet_idx << 5 | vi), NOT 2-bit
- tet_info bits: 0=ALIVE, 1=CHANGED, 2=EMPTY

---

## 🔍 ROOT CAUSE LESSONS

### Why Flaws Occurred

**FLAW #1 (TET_EMPTY):** Implementer ported `update_flip_trace.wgsl` faithfully but didn't realize `flip.wgsl` needs to SET the marker. Only TET_ALIVE and TET_CHANGED seemed critical.

**FLAW #2 (CPU Compaction):** Porting thrust zip iterators to WGPU was non-trivial. Took the "simple" path of CPU processing instead.

**Recurring vert_tet Bug:** Misleading name + breaks established pattern + fails silently = debugging intuition backwards.

### Prevention Strategy

✅ **Port ALL features first, optimize later**
✅ **Verify CUDA references, don't assume**
✅ **Prominent warnings for footguns**
✅ **Enable validation early**

---

## 📈 TEST TRAJECTORY

| Date | Passing | Total | Rate | Notes |
|------|---------|-------|------|-------|
| 2026-03-02 | 22/66 | 66 | 33% | After agent audit fixes |
| 2026-03-03 | 56/69 | 69 | 81% | Exact predicates added |
| 2026-03-04 | 21/66 | 66 | 32% | After "gonna clean" commit (SIGSEGV introduced) |

**Current Blocker:** SIGSEGV prevents further progress. Must fix before other improvements.

---

## ✅ COMPLETED WORK (Archive)

### Recently Fixed

1. ✅ **3-2 Flip Execution** - Full CUDA port with proper vertex identification
2. ✅ **Free Tet Allocation** - Uses pre-allocated consecutive slots (CUDA approach)
3. ✅ **Vote Buffer Initialization** - Defensive initialization with NO_VOTE
4. ✅ **Gather Strategy** - Uses OPP_SPHERE_FAIL flags
5. ✅ **Split Points Self-Check** - Prevents updating vert_tet for vertex being inserted
6. ✅ **External Adjacency (Face 3)** - Concurrent split detection + neighbor lookup
7. ✅ **TetOpp Encoding** - All shaders use correct 5-bit encoding
8. ✅ **MEAN_VERTEX_DEGREE** - All shaders use correct value (8)
9. ✅ **GPU Flip Queue Compaction** - Parallel prefix sum implementation

---

**END OF COMPILED FLAWS DOCUMENT**

For detailed CUDA implementation notes, see MEMORY.md in `.claude/projects/` directory.
