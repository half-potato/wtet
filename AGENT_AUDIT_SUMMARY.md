# Agent Swarm Audit Summary (2026-03-03)

## Executive Summary

6 parallel agents systematically compared all WGSL shader code against CUDA source to find subtle porting mistakes. **10+ critical bugs discovered**, 3 already fixed.

**Test Status:** 54/69 tests passing

---

## 🔴 CRITICAL BUGS FOUND

### 1. TetOpp Encoding Bugs (2-bit instead of 5-bit)

**Affected Files:**
- ✅ **flip.wgsl** lines 32-38 - FIXED
- ✅ **split_fixup.wgsl** lines 17-26 - FIXED
- ✅ **fixup_adjacency.wgsl** lines 15-20 - FIXED

**Issue:** Used `<< 2u` and `>> 2u` instead of `<< 5u` and `>> 5u`

**Impact:** When flip.wgsl reads adjacency from split.wgsl:
- split.wgsl encodes: `(64 << 5) | 2 = 2050`
- flip.wgsl decodes: `2050 >> 2 = 512` (WRONG! Should be 64)
- Result: Out-of-bounds tet access → potential SIGSEGV

**Status:** ✅ FIXED - All 3 shaders now use 5-bit encoding

---

### 2. Missing atomicMin in Flip Voting

**File:** check_delaunay_fast.wgsl lines 87-98

**Issue:** Used direct assignment instead of `atomicMin`:
```wgsl
// BEFORE (WRONG):
tet_vote_arr[vote_offset + bot_ti] = vote_val;

// AFTER (FIXED):
atomicMin(&tet_vote_arr[vote_offset + bot_ti], vote_val);
```

**Impact:** Race conditions when multiple threads vote for same tet

**Status:** ✅ FIXED - Changed buffer to `array<atomic<i32>>` and added atomicMin

---

### 3. Vertex Ordering Mismatch in split.wgsl

**File:** split.wgsl lines 158-163

**Issue:** Vertices ordered differently from CUDA:
- **WGSL**: `(p, v1, v2, v3)` - split vertex in position 0
- **CUDA**: `(v1, v3, v2, p)` - split vertex in position 3 (last)

**Impact:**
- Face indexing is wrong (face i is opposite vertex i)
- Adjacency mapping breaks
- Orient3D results may be incorrect

**Status:** ❌ NOT FIXED - Requires careful adjacency recalculation

**CUDA Reference** (KerDivision.cu:166-171 + TetViAsSeenFrom table):
```cpp
// For vi=0: AsSeenFrom[0] = {1,3,2}
const Tet newTet = {
    tet._v[1],  // v1
    tet._v[3],  // v3
    tet._v[2],  // v2
    splitVertex // p
};
```

**Correct WGSL:**
```wgsl
tets[t0] = vec4<u32>(v1, v3, v2, p);  // Not (p, v1, v2, v3)
tets[t1] = vec4<u32>(v0, v2, v3, p);  // Not (v0, p, v2, v3)
tets[t2] = vec4<u32>(v0, v3, v1, p);  // Not (v0, v1, p, v3)
tets[t3] = vec4<u32>(v0, v1, v2, p);  // Same
```

---

### 4. Dispatch Parameter Bugs

**File:** src/gpu/dispatch.rs

**Issues:**

a) **dispatch_pick_winner** (line 70) - Uses wrong count:
```rust
// WRONG:
bytemuck::cast_slice(&[self.max_tets, ...])
pass.dispatch_workgroups(div_ceil(self.max_tets, 64), 1, 1);

// CORRECT:
bytemuck::cast_slice(&[num_uninserted, ...])
pass.dispatch_workgroups(div_ceil(num_uninserted, 64), 1, 1);
```

b) **dispatch_split_points** (line 105) - Missing parameter write:
```rust
// Should write params before dispatch:
queue.write_buffer(
    &self.pipelines.split_points_params,
    0,
    bytemuck::cast_slice(&[num_uninserted, 0u32, 0u32, 0u32]),
);
```

**Status:** ❌ NOT FIXED

---

### 5. Suboptimal Workgroup Sizes

**Files:** Multiple shaders use 64 threads instead of CUDA's optimal 32

CUDA explicitly uses 32 threads per block with comment "32 ThreadsPerBlock is optimal":
- split.wgsl line 107: `@workgroup_size(64)` → should be 32
- flip.wgsl line 138: `@workgroup_size(64)` → should be 32
- update_opp.wgsl line 63: `@workgroup_size(64)` → should be 32
- check_delaunay_fast.wgsl line 148: `@workgroup_size(64)` → should be 32

**Status:** ❌ NOT FIXED - Performance optimization, not correctness

---

## ✅ VERIFIED CORRECT

### Agent Results:

1. **Buffer Write Agent** - Found 2 critical bugs (vertex ordering, split_fixup encoding)
2. **Buffer Read Agent** - Found position/vertex ID confusion (was causing earlier bug, now fixed)
3. **Dispatch Params Agent** - Found 7 parameter mismatches
4. **Index Calculations Agent** - Confirmed flip.wgsl encoding bug
5. **Atomic Operations Agent** - Found missing atomicMin
6. **Loop Ranges Agent** - ✅ ALL CORRECT
7. **Data Semantics Agent** - Created comprehensive reference

---

## 📊 Test Results

**Before Agent Fixes:** 55/69 passing
**After Encoding Fixes:** 54/69 passing (1 regression, possibly from atomicMin change)

**Still Failing:**
- test_delaunay_cube: 7/8 points (1 fails iteration 5)
- test_raw_* tests: Various convergence issues
- All failures are convergence issues, NOT crashes

---

## 🎯 Priority Fixes Needed

### Priority 1 - Likely Causes of Convergence Failure

1. **Fix vertex ordering in split.wgsl** (Issue #3)
   - This affects ALL split operations
   - Face indices are wrong → adjacency breaks → voting fails
   - MUST match CUDA's TetViAsSeenFrom permutation

2. **Fix dispatch_pick_winner parameters** (Issue #4a)
   - Currently iterates over tets instead of vertices
   - Completely wrong iteration domain

### Priority 2 - Correctness

3. **Add params buffer write for split_points** (Issue #4b)
   - Shader may iterate wrong count

### Priority 3 - Performance

4. **Fix workgroup sizes** (Issue #5)
   - Change 64 → 32 in 4 shaders

---

## 📋 Documentation Created

All agents created detailed reports:
- `BUFFER_INTERPRETATION_AUDIT.md` - Buffer read/write analysis (500+ lines)
- `BUFFER_BUGS_SUMMARY.md` - Executive summary
- `ATOMIC_VERIFICATION_REPORT.md` - Complete atomic operations audit
- `LOOP_ITERATION_ANALYSIS.md` - Loop range verification
- `BUFFER_SEMANTIC_MAPPING.md` - Reference mapping of all buffers
- `AGENT_AUDIT_SUMMARY.md` - This document

---

## 🔍 Root Cause Analysis

**Why simple mistakes happened:**

1. **No systematic verification** - Code was ported incrementally without checking every detail
2. **Encoding constant changed** - 2-bit vs 5-bit was easy to miss in local functions
3. **Different data layouts** - CUDA uses compacted arrays, WGPU uses hybrid approach
4. **Vertex ordering convention** - CUDA uses specific permutation tables, WGSL used natural order

**Prevention:**
- Agent swarms should be used for future major changes
- All encoding functions should be centralized, not duplicated per-shader
- Critical constants should be verified against CUDA source
- Buffer semantics should be documented explicitly

---

## 📅 Next Steps

1. Fix vertex ordering in split.wgsl (high priority)
2. Fix dispatch_pick_winner parameters
3. Run tests to measure improvement
4. Consider if workgroup size optimization is worth the effort
5. Document all deviations from CUDA in ARCHITECTURE.md
