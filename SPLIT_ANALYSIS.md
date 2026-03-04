# split.wgsl Analysis: Encoding Bug and Race Condition

**Date:** 2026-03-02
**Context:** Analysis requested to understand the race condition mentioned in split.wgsl:216 TODO

---

## 🔴 CRITICAL BUG #1: Incorrect TetOpp Encoding

### Issue

**split.wgsl lines 62-72 use 2-bit encoding instead of 5-bit:**

```wgsl
fn encode_opp(tet_idx: u32, face: u32) -> u32 {
    return (tet_idx << 2u) | (face & 3u);  // ❌ WRONG: Should be << 5u
}

fn decode_opp_tet(packed: u32) -> u32 {
    return packed >> 2u;  // ❌ WRONG: Should be >> 5u
}
```

### CUDA Reference

**CommonTypes.h:277-279 (setOppInternal):**
```cpp
__forceinline__ __host__ __device__ void setOppInternal( int vi, int tetIdx, int oppTetVi )
{
    _t[ vi ] = ( tetIdx << 5 ) | ( 1 << 2 ) | oppTetVi;  // ✅ 5-bit shift
}
```

**CommonTypes.h:248-265 (standard TetOpp encoding):**
```cpp
int getOppValTet(int val) { return (val >> 5); }  // ✅ 5-bit shift
void setOppValTet(int &val, int idx) { val = (val & 0x1f) | (idx << 5); }
int getOppValVi(int val) { return (val & 3); }
int makeOppVal(int tetIdx, int oppTetVi) { return (tetIdx << 5) | oppTetVi; }
```

### Impact

These functions are used **15 times** in split.wgsl:
- Lines 192-212: Internal adjacency setup (12 calls to `encode_opp`)
- Lines 225-228: External adjacency update (2 calls to `decode_opp_tet`, 1 to `encode_opp`)

**Result:** All internal adjacency is encoded incorrectly, causing:
- Neighbor lookups extract wrong tet indices (off by factor of 8)
- Adjacency structure corruption
- Point location walks fail
- Insertion failures and crashes

### Fix Required

```wgsl
fn encode_opp(tet_idx: u32, face: u32) -> u32 {
    return (tet_idx << 5u) | (face & 3u);  // ✅ CORRECT
}

fn decode_opp_tet(packed: u32) -> u32 {
    return packed >> 5u;  // ✅ CORRECT
}
```

**Status:** This contradicts MEMORY.md which claims "All shaders verified to use 5-bit encoding (2026-03-02)". The encoding bug was NOT fixed in split.wgsl.

---

## ⚠️ ISSUE #2: Race Condition in External Adjacency Update

### The Problem

**Commented-out code (split.wgsl:217-231):**
```wgsl
// TEMPORARILY DISABLED: This causes race conditions when multiple threads split the same tet
// TODO: Fix voting system to ensure only 1 winner per tet, OR implement atomic claiming
/*
for (var k = 0u; k < 4u; k++) {
    let ext_opp = ext_opps[k];

    if ext_opp != INVALID {
        var nei_tet = decode_opp_tet(ext_opp);
        var nei_face = decode_opp_face(ext_opp);

        set_opp_at(nei_tet, nei_face, encode_opp(new_tets[k], k));
    }
}
*/
```

### Analysis of the TODO Comment

The TODO says: **"Fix voting system to ensure only 1 winner per tet"**

**This is misleading!** The voting system ALREADY ensures only 1 winner per tet:
- `vote.wgsl:63` uses `atomicMin(&tet_vote[tet_idx], vote)`
- Only the closest vertex wins each tet
- Multiple vertices CANNOT split the same tet

### The ACTUAL Race Condition

The race condition is **NOT** about multiple vertices splitting the same tet. It's about:

**When two neighboring tets split simultaneously:**

1. Tet A splits and tries to update neighbor B's adjacency
2. Tet B splits and tries to update neighbor A's adjacency
3. **RACE:** Both threads write to each other's adjacency arrays concurrently
4. **Corruption:** Each overwrites the other's internal adjacency with stale external references

**Example:**
```
Initial state:     A ↔ B
After A splits:    A0,A1,A2,A3 → B (A sets B's adjacency)
After B splits:    A ← B0,B1,B2,B3 (B sets A's adjacency - but A no longer exists!)
```

### How CUDA Solves This

**KerDivision.cu:140-162 (kerSplitTetra):**

```cpp
// Get neighbor's tet index and face
int neiTetIdx = oldOpp.getOppTet( vi );
int neiTetVi  = oldOpp.getOppVi( vi );

// Check if neighbour has split
const int neiSplitIdx = tetToVert[ neiTetIdx ];

if ( neiSplitIdx == INT_MAX ) // Neighbour is un-split
{
    // Safe to update: neighbor hasn't been modified yet
    oppArr[ neiTetIdx ].setOpp( neiTetVi, newTetIdx[ vi ], 3 );
}
else // Neighbour has split
{
    // Neighbor is splitting concurrently - use lookup to find correct new tet
    const int neiSplitVert  = vertArr[ neiSplitIdx ];
    const int neiFreeIdx    = ( neiSplitVert + 1 ) * MeanVertDegree - 1;

    // Use free_arr to find which of neighbor's 4 split tets has this face
    neiTetIdx = freeArr[ neiFreeIdx - neiTetVi ];
    neiTetVi  = 3;  // All external faces become face 3 after split
}

// Point this tetra to neighbour (correct tet, whether split or not)
newOpp.setOpp( 3, neiTetIdx, neiTetVi );
```

**Key insight:** CUDA uses `tetToVert` to detect if a neighbor is splitting, then uses `free_arr` indexing to find the correct split tet.

### Why WGSL Code is Disabled

The commented-out WGSL code assumes neighbors are never split:
```wgsl
set_opp_at(nei_tet, nei_face, encode_opp(new_tets[k], k));
```

This is equivalent to only the `if ( neiSplitIdx == INT_MAX )` branch in CUDA - missing the `else` case!

**Result:**
- Without the check, both threads corrupt each other's adjacency
- Adjacency structure becomes invalid
- Subsequent operations (flips, insertions) crash or fail
- Code was disabled to prevent crashes, but this breaks correctness

---

## 🔧 Required Fixes

### Fix #1: Correct TetOpp Encoding (CRITICAL)

**File:** `src/shaders/split.wgsl`
**Lines:** 62-72

```wgsl
fn encode_opp(tet_idx: u32, face: u32) -> u32 {
    return (tet_idx << 5u) | (face & 3u);  // Changed: 2u → 5u
}

fn decode_opp_tet(packed: u32) -> u32 {
    return packed >> 5u;  // Changed: 2u → 5u
}

fn decode_opp_face(packed: u32) -> u32 {
    return packed & 3u;  // Already correct
}
```

**Also update MEMORY.md** to reflect that split.wgsl was NOT fixed.

### Fix #2: Implement Concurrent Split Detection (HIGH PRIORITY)

**File:** `src/shaders/split.wgsl`
**Lines:** 214-231 (uncomment and fix)

Replace the commented-out code with CUDA's logic:

```wgsl
// Update external neighbours to point back to us
let ext_opps = array<u32, 4>(orig_opp_0, orig_opp_1, orig_opp_2, orig_opp_3);
let new_tets = array<u32, 4>(t0, t1, t2, t3);

for (var k = 0u; k < 4u; k++) {
    let ext_opp = ext_opps[k];

    if ext_opp != INVALID {
        var nei_tet = decode_opp_tet(ext_opp);
        var nei_face = decode_opp_face(ext_opp);

        // Check if neighbor is splitting concurrently
        let nei_split_idx = tet_to_vert[nei_tet];

        if nei_split_idx != INVALID {
            // Neighbor has split - use free_arr to find correct new tet
            let nei_split_vert = insert_list[nei_split_idx].y;  // vertex being inserted
            let nei_free_idx = (nei_split_vert + 1u) * MEAN_VERTEX_DEGREE - 1u;

            nei_tet = free_arr[nei_free_idx - nei_face];
            nei_face = 3u;  // External faces become face 3
        }

        // Point neighbor back to this new tet (safe whether split or not)
        set_opp_at(nei_tet, nei_face, encode_opp(new_tets[k], k));
    }
}
```

**Also update TODO comment:**
```wgsl
// Update external neighbours to point back to us.
// Uses tet_to_vert to detect concurrent splits (CUDA: KerDivision.cu:140-162)
```

---

## 📋 Summary

| Issue | Severity | Status | Fix Required |
|-------|----------|--------|--------------|
| **2-bit encoding instead of 5-bit** | 🔴 CRITICAL | Unfixed | Change `<< 2u` → `<< 5u` and `>> 2u` → `>> 5u` |
| **External adjacency disabled** | ⚠️ HIGH | Disabled (broken) | Implement concurrent split detection using `tet_to_vert` |
| **Misleading TODO comment** | 📝 Documentation | Active | Replace with accurate description of race condition |

**Dependencies:**
- `tet_to_vert` buffer already exists (binding 9) ✅
- `free_arr` buffer already exists (binding 5) ✅
- `insert_list` buffer already exists (binding 4) ✅
- All required data is available to implement the fix

**Impact of current bugs:**
1. **2-bit encoding:** Completely breaks adjacency structure → insertion failures, crashes
2. **Disabled external updates:** Adjacency is incomplete → flip detection may miss cases, incorrect Delaunay triangulation

**Priority:** Fix encoding bug FIRST (blocks all functionality), then implement concurrent split detection.
