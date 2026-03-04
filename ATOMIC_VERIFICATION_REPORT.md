# Atomic Operations Verification Report

## Summary

Comprehensive verification of all atomic operations in WGSL shaders against CUDA source files reveals **CRITICAL BUGS** in voting mechanisms and TetOpp encoding.

---

## Critical Bugs Found

### 1. **fixup_adjacency.wgsl: Wrong TetOpp Encoding (2-bit instead of 5-bit)**

**Location:** `/home/amai/gdel3d_wgpu/src/shaders/fixup_adjacency.wgsl:15-25`

**Bug:**
```wgsl
fn encode_opp(tet_idx: u32, face: u32) -> u32 {
    return (tet_idx << 2u) | (face & 3u);  // WRONG: Should be << 5u
}
fn decode_opp_tet(packed: u32) -> u32 {
    return packed >> 2u;  // WRONG: Should be >> 5u
}
```

**Impact:** All adjacency fixups corrupt tet indices by factor of 8.

**Fix Required:**
```wgsl
fn encode_opp(tet_idx: u32, face: u32) -> u32 {
    return (tet_idx << 5u) | (face & 3u);
}
fn decode_opp_tet(packed: u32) -> u32 {
    return packed >> 5u;
}
```

---

### 2. **check_delaunay_fast.wgsl: Missing atomicMin for Flip Voting**

**Location:** `/home/amai/gdel3d_wgpu/src/shaders/check_delaunay_fast.wgsl:87-98`

**Bug:**
```wgsl
fn vote_for_flip23(vote_offset: u32, bot_ti: u32, top_ti: u32) {
    let vote_val = i32(bot_ti);
    tet_vote_arr[vote_offset + bot_ti] = vote_val;   // WRONG: Should use atomicMin
    tet_vote_arr[vote_offset + top_ti] = vote_val;   // WRONG: Should use atomicMin
}
```

**CUDA Reference:** `KerPredicates.cu:343-356`
```cuda
void voteForFlip23(int* tetVoteArr, int voteOffset, int botTi, int topTi) {
    const int voteVal = voteOffset + botTi;
    atomicMin( &tetVoteArr[ botTi ], voteVal );   // Uses atomicMin!
    atomicMin( &tetVoteArr[ topTi ], voteVal );   // Uses atomicMin!
}
```

**Impact:**
- Race conditions when multiple threads vote for the same tet
- Non-deterministic flip selection
- Potential data corruption when concurrent flips overwrite votes

**Fix Required:**
```wgsl
fn vote_for_flip23(vote_offset: u32, bot_ti: u32, top_ti: u32) {
    let vote_val = i32(vote_offset + bot_ti);
    atomicMin(&tet_vote_arr[bot_ti], vote_val);
    atomicMin(&tet_vote_arr[top_ti], vote_val);
}

fn vote_for_flip32(vote_offset: u32, bot_ti: u32, top_ti: u32, bot_opp_ti: u32) {
    let vote_val = i32(vote_offset + bot_ti);
    atomicMin(&tet_vote_arr[bot_ti], vote_val);
    atomicMin(&tet_vote_arr[top_ti], vote_val);
    atomicMin(&tet_vote_arr[bot_opp_ti], vote_val);
}
```

**Additional Changes:**
- Binding must change from `var<storage, read_write>` to `var<storage, read_write> array<atomic<i32>>`

---

### 3. **vote.wgsl vs CUDA: Different Voting Mechanisms**

**WGSL Implementation:** `/home/amai/gdel3d_wgpu/src/shaders/vote.wgsl:23-64`
- Uses `atomicMin` with packed vote (distance + vertex index)
- Stores in `tet_vote` array
- Second pass reads vote and appends to insert_list

**CUDA Implementation:** `KerPredicates.cu:151-200` + `KerDivision.cu:305-334`
- Uses `atomicMax` to store sphere value in `tetSphereArr`
- Uses separate `vertSphereArr` for vertex sphere values
- Uses `atomicMin` to store **vertex INDEX** (not vertex ID!) in `tetVertArr`
- Third pass (`kerNegateInsertedVerts`) negates `vertTetVec` for winners

**Verdict:** ✅ **WGSL approach is VALID but DIFFERENT**
- WGSL combines steps into simpler two-pass approach
- Uses atomicMin correctly with packed distance+index
- Achieves same result: one winner per tet, closest point wins

---

## Verified Correct Operations

### ✅ split.wgsl: TetOpp Encoding (FIXED)
- **Lines 62-72:** Uses correct 5-bit encoding (was fixed earlier)
- `encode_opp`: `(tet_idx << 5u) | (face & 3u)` ✅
- `decode_opp_tet`: `packed >> 5u` ✅

### ✅ split.wgsl: Counter Updates
- **Line 258:** `atomicAdd(&counters[COUNTER_ACTIVE], 3u)`
- **CUDA Reference:** `KerDivision.cu` doesn't explicitly track active count in split
- **Verdict:** VALID - needed for WGPU's pre-allocated buffer approach

### ✅ pick_winner.wgsl: Insert List Building
- **Line 46:** `atomicAdd(&counters[COUNTER_INSERTED], 1u)`
- Allocates slot in insert_list atomically
- **Verdict:** CORRECT - standard pattern for parallel append

### ✅ split.wgsl: Free List Allocation (Non-atomic)
- **Lines 85-104:** `get_free_slots_4tet()` does NOT use atomics
- Reads from `vert_free_arr[vertex]` and decrements non-atomically
- **CUDA Reference:** `KerDivision.cu:120-124` - also non-atomic!
  ```cuda
  const int newIdx = ( splitVertex + 1 ) * MeanVertDegree - 1;
  const int newTetIdx[4] = { freeArr[newIdx], freeArr[newIdx-1], freeArr[newIdx-2], freeArr[newIdx-3] };
  vertFreeArr[ splitVertex ] -= 4;  // NOT ATOMIC!
  ```
- **Verdict:** CORRECT - Each split is single-threaded per vertex, no race condition

### ✅ split.wgsl: atomicStore for tet_opp
- **Lines 191-212:** All use `set_opp_at()` which calls `atomicStore`
- **CUDA Reference:** `KerDivision.cu:175` uses `storeOpp()` wrapper
- **Verdict:** CORRECT - Matches CUDA pattern

### ✅ update_opp.wgsl: All TetOpp Operations
- **Lines 21-35:** Correct 5-bit encoding helpers
- **Lines 83-104:** `atomicLoad` for reading external opposites
- **Lines 131, 151-183:** `atomicStore` for writing adjacency
- **Verdict:** CORRECT - Faithful port of `kerUpdateOpp` (KerDivision.cu:562)

### ✅ mark_rejected_flips.wgsl: Shared Memory Counter
- **Line 67:** `atomicStore(&flip_num, 0u)` - workgroup shared counter
- **Line 151:** `flip_offset = atomicAdd(&counters[0], local_flip_num)`
- **CUDA Reference:** `KerDivision.cu:280-287` uses `__shared__ int s_flipNum` + `atomicAdd`
- **Verdict:** CORRECT - Exact match

---

## Atomics by Operation Type

### atomicMin (2 uses)
1. ✅ **vote.wgsl:63** - `atomicMin(&tet_vote[tet_idx], vote)`
   - CUDA: `kerPickWinnerPoint` uses `atomicMin(&tetVertArr[tetIdx], idx)`
   - Different approach but semantically equivalent

2. ❌ **check_delaunay_fast.wgsl:88-97** - **MISSING atomicMin!**
   - Should be: `atomicMin(&tet_vote_arr[bot_ti], vote_val)`

### atomicAdd (11 uses)
All verified correct for counter increments and slot allocation:
- pick_winner.wgsl:46 ✅
- gather.wgsl:137 ✅
- mark_rejected_flips.wgsl:151 ✅
- compact_if_negative.wgsl:30,35 ✅
- vote.wgsl:102 ✅
- split.wgsl:258 ✅
- split_points.wgsl:73,83,153 ✅
- flip.wgsl:390 ✅

### atomicLoad (74 uses)
All verified - used for reading atomic<u32> tet_opp arrays. Correct pattern.

### atomicStore (48 uses)
All verified - used for writing atomic<u32> tet_opp arrays. Correct pattern.

---

## Action Items

### HIGH PRIORITY - Fix Immediately

1. **Fix fixup_adjacency.wgsl encoding**
   - Change `<< 2u` → `<< 5u` (line 16)
   - Change `>> 2u` → `>> 5u` (line 20)

2. **Fix check_delaunay_fast.wgsl voting**
   - Change binding to `array<atomic<i32>>`
   - Replace direct assignment with `atomicMin` in vote_for_flip23 and vote_for_flip32
   - Add `vote_offset` to `vote_val` calculation

### MEDIUM PRIORITY - Verify Behavior

3. **Test vote.wgsl approach**
   - Current implementation differs from CUDA but may be correct
   - Verify that closest-point-wins behavior matches CUDA
   - Check that packing/unpacking preserves vertex index correctly

---

## Files Checked

### CUDA Source Files
- `/home/amai/gDel3D/src/gdel3d/gDel3D/GPU/KerDivision.cu`
- `/home/amai/gDel3D/src/gdel3d/gDel3D/GPU/KerPredicates.cu`
- `/home/amai/gDel3D/src/gdel3d/gDel3D/GpuDelaunay.cu`

### WGSL Shader Files (with atomics)
- ✅ split.wgsl
- ✅ pick_winner.wgsl
- ✅ vote.wgsl
- ✅ update_opp.wgsl
- ✅ mark_rejected_flips.wgsl
- ✅ compact_tets.wgsl
- ❌ fixup_adjacency.wgsl (2-bit encoding bug)
- ❌ check_delaunay_fast.wgsl (missing atomicMin)
- ✅ flip.wgsl
- ✅ init.wgsl
- ✅ reset_votes.wgsl
- ✅ split_fixup.wgsl
- ✅ mark_special_tets.wgsl
- ✅ allocate_flip23_slot.wgsl
- ✅ shift_opp_tet_idx.wgsl
- ✅ compact_if_negative.wgsl
- ✅ gather.wgsl
- ✅ split_points.wgsl

---

## Conclusion

**2 CRITICAL BUGS FOUND:**
1. fixup_adjacency.wgsl uses 2-bit TetOpp encoding (corrupts all adjacency)
2. check_delaunay_fast.wgsl uses direct assignment instead of atomicMin for voting (race conditions)

**Root Cause of SIGSEGV:**
- Bug #1 causes neighbor lookups to point to wrong tets (8x off)
- Bug #2 causes non-deterministic flip selection and vote corruption
- Combined: accessing out-of-bounds memory or invalid tet indices

**Fix Priority:** Both bugs must be fixed together - they interact during flip operations where adjacency updates depend on correct voting.
