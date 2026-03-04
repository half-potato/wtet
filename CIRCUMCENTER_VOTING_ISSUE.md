# Circumcenter Voting Issue

**Date:** 2026-03-04
**Status:** ❌ REVERTED - Needs debugging

## What We Tried

Implemented circumcenter-based voting to match CUDA's `InsCircumcenter` mode:

1. ✅ Added `insphere_fast()` to compute insphere determinant
2. ✅ Clamped negative values to 0 (CUDA: KerPredicates.cu:186-187)
3. ✅ Used bitcast to convert float to u32 for voting
4. ✅ Changed to `atomicMax` (CUDA uses atomicMax, line 196)
5. ✅ Changed NO_VOTE to minimum i32 (-2147483648)

## The Bug

**Symptom:** Way too many insertions per iteration
- Expected: 1-4 points per iteration
- Actual: 16, 24, 28+ points per iteration
- Test: test_raw_cospherical_12 with 12 points tried to insert 16 in iteration 4

**Error:**
```
Copy of 0..128 would end up overrunning the bounds of the Source buffer of size 96
```

## Possible Root Causes

### 1. Vote Packing Issue

**Current packing:**
```wgsl
fn pack_vote(sphere_val_bits: u32, local_idx: u32) -> i32 {
    return i32((sphere_val_bits << 16u) | (local_idx & 0xFFFFu));
}
```

**Problem:** When sphere_val is 0.0 (clamped negative), `bitcast<u32>(0.0) = 0x00000000`.
- All points inside sphere get same vote: `(0x00000000 << 16) | idx = idx`
- atomicMax picks highest index, not random/first
- This might cause duplicate votes or wrong vote interpretation

### 2. NO_VOTE Check

**Current check (pick_winner_point):**
```wgsl
if vote == NO_VOTE {
    return;
}
```

**Problem:** With atomicMax and NO_VOTE = -2147483648:
- Any positive vote > -2147483648
- But vote = 0 (all points inside) should perhaps be rejected?

### 3. Float-to-Int Bitcast Ordering

**CUDA uses:** `__float_as_int(sval)` for atomicMax
**WGSL uses:** `bitcast<u32>(sphere_val)` then cast to i32

**Problem:** IEEE 754 float ordering when treated as signed int:
- Negative floats: high bit set, map to negative ints (WRONG order!)
- Positive floats: correctly ordered as unsigned, but need sign handling

**Solution needed:** Flip sign bit for correct signed ordering
```wgsl
// Correct approach for atomicMax with floats:
var bits = bitcast<u32>(sphere_val);
if sphere_val < 0.0 {
    bits = bits ^ 0x80000000u;  // Flip sign bit
} else {
    bits = bits | 0x80000000u;  // Set sign bit for positive
}
```

### 4. Voter Index vs Vertex Index Confusion

**pack_vote takes `local_idx`** which is the index into the uninserted array.
**unpack_vote_idx** returns this local_idx.
**pick_winner_point** then does: `let vert_idx = uninserted2[local_idx];`

**If local_idx is wrong** (e.g., corrupted vote value), this could index out of bounds or pick wrong vertices.

## CUDA Reference

**kerVoteForPoint** (KerPredicates.cu:160-200):
```cpp
// Line 174: Compute sphere value
sval = dPredWrapper.inSphereDet( tet, vert );

// Line 186-187: Clamp negative
if ( sval < 0 )
    sval = 0;

// Line 189: Convert to int bits
int ival = __float_as_int(sval);

// Line 191: Store in array
vertSphereArr[ idx ] =  ival;

// Line 195-196: Vote with atomicMax
if ( tetSphereArr[ tetIdx ] < ival )
    atomicMax( &tetSphereArr[ tetIdx ], ival );
```

**Key difference:** CUDA stores sphere values in separate array (`vertSphereArr`), then votes with raw int-as-float values. WGPU packs sphere value + index into single i32.

## Workaround

Reverted to centroid-based voting with atomicMin:
- Uses simple distance^2 to centroid
- Quantized to 16-bit key
- Packed with vertex index in lower 16 bits
- atomicMin picks closest point
- NO_VOTE = 0x7FFFFFFF (maximum i32)

## To Fix

Need to either:

1. **Match CUDA structure:** Separate sphere value array + index array
   - Store sphere values in tet_vote
   - Store indices in separate array
   - Pick winner reads both arrays

2. **Fix vote packing for atomicMax:**
   - Properly handle float-to-int bitcast for signed comparison
   - Flip sign bit for negative floats
   - Ensure 0.0 sphere values don't create duplicate votes

3. **Use different voting metric:**
   - Keep atomicMin with distance to circumcenter instead of determinant
   - Compute actual distance instead of insphere sign

## Files Modified (Then Reverted)

- `src/shaders/vote.wgsl` - Changed voting logic (REVERTED)
- `src/shaders/reset_votes.wgsl` - Changed NO_VOTE constant (REVERTED)

## Current Status

✅ Reverted to working state (centroid + atomicMin)
❌ Circumcenter voting not implemented
📝 Need to debug vote packing before retry
