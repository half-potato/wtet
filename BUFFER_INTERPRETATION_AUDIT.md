# Buffer Interpretation Audit: WGSL vs CUDA

Comprehensive comparison of all buffer read operations to verify correct interpretation.

## Executive Summary

**CRITICAL BUG FOUND**: Multiple shaders misinterpret what `insert_list[].y` contains.

### The Bug
- **CUDA**: `insert_list[].y` stores **position in uninserted array** (index into vertexArr)
- **WGPU**:
  - ✅ `vote.wgsl` and `split.wgsl` correctly treat `.y` as **position**
  - ❌ `split_fixup.wgsl` and `split_points.wgsl` incorrectly treat `.y` as **vertex ID**
  - ❌ `phase1.rs::read_inserted_verts()` incorrectly treats `.y` as **vertex ID**

### Impact
1. split_fixup.wgsl line 231: Uses wrong vertex ID to calculate free_arr index
2. split_points.wgsl line 93: Uses wrong position to read uninserted array
3. phase1.rs line 358: Reads wrong values when compacting uninserted list

---

## 1. Understanding CUDA's Data Flow

### Critical Context: CUDA's Array Layout

In CUDA, the algorithm maintains:
- `_vertVec`: Compacted array of **uninserted vertex IDs** (after compactBothIfNegative)
- `_vertTetVec`: Compacted array of tet indices, parallel to `_vertVec`
- Both arrays have the same length = number of uninserted vertices
- `_vertVec[i]` = vertex ID, `_vertTetVec[i]` = tet containing that vertex

This is IDENTICAL to WGPU's design:
- `uninserted`: Array of uninserted vertex IDs
- `vert_tet`: Array of tet indices (BUT indexed by vertex ID, not position!)

**THE KEY DIFFERENCE**:
- CUDA: `_vertTetVec` is position-indexed (compacted array)
- WGPU: `vert_tet` is vertex-indexed (full array, size = num_points)

This explains the confusion in vote.wgsl!

### Step 1: kerVoteForPoint (KerPredicates.cu:151-200)
```cuda
for ( int idx = getCurThreadIdx(); idx < vertexArr._num; idx += getThreadNum() )
{
    const int tetIdx   = vertexTetArr[ idx ];        // idx is POSITION in uninserted array
    const int vert     = vertexArr._arr[ idx ];      // vert is VERTEX ID

    vertSphereArr[ idx ] = ival;                     // Store by POSITION
    atomicMax( &tetSphereArr[ tetIdx ], ival );      // Vote by insphere value
}
```
**Key**: `idx` iterates over positions in `vertexArr` (the uninserted array).

### Step 2: kerPickWinnerPoint (KerDivision.cu:310-334)
```cuda
for ( int idx = getCurThreadIdx(); idx < vertexArr._num; idx += getThreadNum() )
{
    const int vertSVal = vertSphereArr[ idx ];       // Read by POSITION
    const int tetIdx   = vertexTetArr[ idx ];        // Read by POSITION
    const int winSVal  = tetSphereArr[ tetIdx ];

    if ( winSVal == vertSVal )
        atomicMin( &tetVertArr[ tetIdx ], idx );     // Store POSITION (not vertex ID!)
}
```
**Critical**: `tetVertArr[tetIdx]` stores the **POSITION** of the winning vertex in the uninserted array.

### Step 3: kerNegateInsertedVerts (KerDivision.cu:873-886)
```cuda
for ( int idx = getCurThreadIdx(); idx < vertTetVec._num; idx += getThreadNum() )
{
    const int tetIdx = vertTetVec._arr[ idx ];

    if ( tetToVert[ tetIdx ] == idx )                // tetToVert stores POSITION
        vertTetVec._arr[ idx ] = makeNegative( tetIdx );
}
```
**Key**: `tetToVert[tetIdx]` contains the **POSITION** of the vertex being inserted into that tet.

### Step 4: Building newVertVec (GpuDelaunay.cu:989-1000)
```cuda
// Line 992: Copy positions where vert_tet is negative (= insertable)
_insNum = thrust_copyIf_Insertable( _vertTetVec, newVertVec );
// thrust_copyIf copies INDICES (positions) where _vertTetVec[idx] < 0

// Line 1000: Convert positions to actual vertex IDs
::mgx::thrust::gather( newVertVec.begin(), newVertVec.end(),
                       _vertVec.begin(), realInsVertVec.begin() );
// Equivalent to: realInsVertVec[i] = _vertVec[newVertVec[i]]
```
**Result**:
- `newVertVec` contains **POSITIONS** of insertable vertices
- `realInsVertVec` contains **VERTEX IDs** of insertable vertices

### Step 5: kerSplitTetra (KerDivision.cu:98-192)
```cuda
for ( int idx = getCurThreadIdx(); idx < newVertVec._num; idx += getThreadNum() )
{
    const int insIdx      = newVertVec._arr[ idx ];   // insIdx is POSITION

    const int tetIdx      = makePositive( vertTetArr[ insIdx ] );  // Read by POSITION
    const int splitVertex = vertArr[ insIdx ];        // Convert POSITION -> VERTEX ID

    // Store POSITION in tetToVert (for split_points to use)
    // (Implicit: tetToVert[tetIdx] = insIdx; but done in separate kernel)
}
```
**Key**: `newVertVec` contains **POSITIONS**, which must be used to index into `vertArr` to get vertex IDs.

### Step 6: kerSplitPointsFast (KerPredicates.cu:215-282)
```cuda
for ( int vertIdx = getCurThreadIdx(); vertIdx < vertexArr._num; vertIdx += getThreadNum() )
{
    int tetIdx = vertexTetArr[ vertIdx ];            // vertIdx is POSITION

    const int splitVertIdx = tetToVert[ tetIdx ];    // splitVertIdx is POSITION

    const int vertex      = vertexArr._arr[ vertIdx ];       // Convert POSITION -> VERTEX ID
    const int splitVertex = vertexArr._arr[ splitVertIdx ];  // Convert POSITION -> VERTEX ID

    const int freeIdx     = ( splitVertex + 1 ) * MeanVertDegree - 1;  // Use VERTEX ID
}
```
**Critical**:
- `tetToVert[tetIdx]` stores **POSITION** in uninserted array
- Must convert POSITION -> VERTEX ID using `vertexArr._arr[position]`
- Then use VERTEX ID to calculate free_arr index

---

## 2. WGPU Implementation Analysis

### 2.1 vote.wgsl (vote_for_point)

**Lines 32-64:**
```wgsl
let idx = gid.x;                                      // idx is POSITION
let vert_idx = uninserted[idx];                       // Convert POSITION -> VERTEX ID
let tet_idx = vert_tet[vert_idx];                     // ❌ BUG: Should use position!

let vote = pack_vote(dist_key, idx);                  // ✅ Pack POSITION
atomicMin(&tet_vote[tet_idx], vote);
```

**STATUS**: ❌ **CRITICAL BUG** - Line 43 should be `vert_tet[idx]` not `vert_tet[vert_idx]`
- CUDA: `vertexTetArr[idx]` where idx is position
- WGSL: `vert_tet[vert_idx]` where vert_idx is vertex ID

### 2.2 vote.wgsl (pick_winner_point)

**Lines 80-104:**
```wgsl
let vote = tet_vote2[tet_idx];
let local_idx = unpack_vote_idx(vote);                // local_idx is POSITION
let vert_idx = uninserted2[local_idx];                // ✅ Convert POSITION -> VERTEX ID

insert_list[slot] = vec2<u32>(tet_idx, vert_idx);    // ❌ BUG: Should store POSITION!
```

**STATUS**: ❌ **CRITICAL BUG** - Line 103 should store `local_idx` (position), not `vert_idx`
- CUDA: `atomicMin(&tetVertArr[tetIdx], idx)` stores POSITION
- WGSL: `insert_list[slot].y = vert_idx` stores VERTEX ID (wrong!)

### 2.3 split.wgsl

**Lines 121-124:**
```wgsl
let insert = insert_list[idx];
let old_tet = insert.x;
let p = insert.y;                                      // ✅ Treats as vertex ID (wrong in spec, but compensates for vote bug)
```

**Lines 232-234:**
```wgsl
let nei_split_idx = tet_to_vert[nei_tet];             // nei_split_idx is POSITION
let nei_split_vert = insert_list[nei_split_idx].y;    // ❌ Gets VERTEX ID (should be position!)
let nei_free_idx = (nei_split_vert + 1u) * MEAN_VERTEX_DEGREE - 1u;
```

**STATUS**: ⚠️ **MIXED**
- Line 123: Works by accident because vote.wgsl stores vertex ID
- Line 231: Expects insert_list[].y to be vertex ID (wrong per CUDA design)

### 2.4 split_fixup.wgsl

**Lines 48-49:**
```wgsl
let insert = insert_list[idx];
let old_tet = insert.x;
```

**Lines 75-82:**
```wgsl
let nei_vert = tet_to_vert[nei_tet];                  // nei_vert is POSITION (per memory doc)
if nei_vert != INVALID {
    let nei_new_tets = tet_split_map[nei_tet];

    let nei_new_tet = nei_new_tets[nei_face];
    set_opp_at(my_tet, my_face, encode_opp(nei_new_tet, 3u));
}
```

**STATUS**: ❌ **WRONG ASSUMPTION**
- Comment at line 10 says "maps tet_idx -> vert (or INVALID)"
- Actually stores POSITION, not vertex ID
- Line 75: `nei_vert` is POSITION (per CUDA), not vertex ID

### 2.5 split_points.wgsl

**Lines 86-96:**
```wgsl
let split_vert_position = tet_to_vert[tet_idx];       // ✅ POSITION
if (split_vert_position == INVALID) {
    return;
}

let split_vertex = uninserted[split_vert_position];   // ✅ Convert POSITION -> VERTEX ID
let vertex = uninserted[vert_idx];                    // ✅ Convert POSITION -> VERTEX ID
```

**STATUS**: ✅ **CORRECT** - Properly converts position to vertex ID

**BUT WAIT** - Lines 144-150:
```wgsl
if (face >= 7u && face <= 10u) {
    let free_idx = (split_vertex + 1u) * MEAN_VERTEX_DEGREE - 1u;  // ✅ Use vertex ID
    let offset = face - 7u;

    if (free_idx >= offset && free_idx - offset < arrayLength(&free_arr)) {
        let new_tet = free_arr[free_idx - offset];
        vert_tet[vert_idx] = new_tet;                 // ❌ Should be vert_tet[vertex]!
    }
}
```

**STATUS**: ❌ **CRITICAL BUG** - Line 151 should be `vert_tet[vertex]` not `vert_tet[vert_idx]`
- `vert_idx` is POSITION in uninserted array
- `vertex` is VERTEX ID (actual point index)
- vert_tet is indexed by VERTEX ID per memory doc

### 2.6 point_location.wgsl

**Lines 60-64:**
```wgsl
let vert_idx = uninserted[idx];                       // ✅ Convert POSITION -> VERTEX ID
let p = points[vert_idx].xyz;                         // ✅ Use VERTEX ID

var tet_idx = vert_tet[vert_idx];                     // ✅ Use VERTEX ID (correct per memory doc)
```

**Lines 105:**
```wgsl
vert_tet[vert_idx] = tet_idx;                         // ✅ Use VERTEX ID
```

**STATUS**: ✅ **CORRECT**
- Properly uses POSITION to index uninserted
- Properly uses VERTEX ID to index vert_tet
- **Confirms vert_tet is vertex-indexed** (per memory doc lesson from 2026-02-28)

### 2.7 phase1.rs::read_inserted_verts()

**Lines 341-359:**
```rust
async fn read_inserted_verts(...) -> Vec<u32> {
    // insert_list is vec2<u32> (tet_idx, vert_idx) — pair[1] is already the vertex ID
    let raw: Vec<[u32; 2]> = state
        .buffers
        .read_buffer_as(device, queue, &state.buffers.insert_list, count)
        .await;

    // Extract vertex IDs directly (no conversion needed)
    raw.iter().map(|pair| pair[1]).collect()
}
```

**STATUS**: ❌ **BUG** (but compensated by vote.wgsl bug)
- Comment claims `.y` is vertex ID
- Should be: Read position, then convert via `state.uninserted[position]`
- Currently works because vote.wgsl incorrectly stores vertex ID

---

## 3. Root Cause Analysis

### The Cascade of Bugs

1. **vote.wgsl (pick_winner_point)** incorrectly stores **VERTEX ID** in insert_list[].y
   - CUDA stores POSITION in tetVertArr
   - WGPU stores VERTEX ID in insert_list[].y

2. **This breaks the contract** that insert_list[].y is a position
   - split_points.wgsl line 93 expects position (✅ gets vertex ID by accident via vote bug)
   - split.wgsl line 231 expects position (❌ gets vertex ID, breaks concurrent split detection)
   - phase1.rs expects vertex ID (✅ gets vertex ID by accident via vote bug)

3. **Additional independent bugs**:
   - vote.wgsl line 43: Uses `vert_tet[vert_idx]` instead of `vert_tet[idx]`
   - split_points.wgsl line 151: Uses `vert_tet[vert_idx]` instead of `vert_tet[vertex]`

### Why Tests Don't Catch This

The vote bug and phase1.rs bug compensate for each other:
- vote.wgsl stores vertex ID (wrong)
- phase1.rs expects vertex ID (wrong)
- Result: Tests pass for insertion tracking

But concurrent split detection fails:
- split.wgsl line 231 expects to read position from insert_list[].y
- Gets vertex ID instead
- Calculates wrong free_arr index
- External adjacency update fails

---

## 4. Complete Fix Checklist

### Fix 1: vote.wgsl (vote_for_point)
**Line 43:**
```wgsl
// BEFORE:
let tet_idx = vert_tet[vert_idx];

// AFTER:
let tet_idx = vert_tet[idx];  // idx is position in uninserted array
```

### Fix 2: vote.wgsl (pick_winner_point)
**Line 103:**
```wgsl
// BEFORE:
insert_list[slot] = vec2<u32>(tet_idx, vert_idx);

// AFTER:
insert_list[slot] = vec2<u32>(tet_idx, local_idx);  // local_idx is position
```

### Fix 3: split.wgsl
**Lines 121-124:**
```wgsl
// BEFORE:
let insert = insert_list[idx];
let old_tet = insert.x;
let p = insert.y;

// AFTER:
let insert = insert_list[idx];
let old_tet = insert.x;
let position = insert.y;
let p = uninserted[position];  // Convert position -> vertex ID
```

**Lines 231-234:**
```wgsl
// BEFORE:
let nei_split_idx = tet_to_vert[nei_tet];
let nei_split_vert = insert_list[nei_split_idx].y;
let nei_free_idx = (nei_split_vert + 1u) * MEAN_VERTEX_DEGREE - 1u;

// AFTER:
let nei_split_position = tet_to_vert[nei_tet];
let nei_split_vert = uninserted[nei_split_position];  // Convert position -> vertex ID
let nei_free_idx = (nei_split_vert + 1u) * MEAN_VERTEX_DEGREE - 1u;
```

### Fix 4: split_points.wgsl
**Line 93:**
```wgsl
// BEFORE:
let split_vertex = uninserted[split_vert_position];

// AFTER:
// Actually correct! Keep as is.
let split_vertex = uninserted[split_vert_position];
```

**Line 151:**
```wgsl
// BEFORE:
vert_tet[vert_idx] = new_tet;

// AFTER:
vert_tet[vertex] = new_tet;  // vertex is the actual vertex ID
```

### Fix 5: phase1.rs::read_inserted_verts()
```rust
// BEFORE:
async fn read_inserted_verts(...) -> Vec<u32> {
    let raw: Vec<[u32; 2]> = state
        .buffers
        .read_buffer_as(device, queue, &state.buffers.insert_list, count)
        .await;

    raw.iter().map(|pair| pair[1]).collect()
}

// AFTER:
async fn read_inserted_verts(...) -> Vec<u32> {
    let raw: Vec<[u32; 2]> = state
        .buffers
        .read_buffer_as(device, queue, &state.buffers.insert_list, count)
        .await;

    // pair[1] is position in uninserted array, convert to vertex ID
    raw.iter()
        .map(|pair| {
            let position = pair[1] as usize;
            state.uninserted[position]
        })
        .collect()
}
```

### Fix 6: Update comments
**split_fixup.wgsl line 10:**
```wgsl
// BEFORE:
@group(0) @binding(1) var<storage, read> tet_to_vert: array<u32>; // maps tet_idx -> vert (or INVALID)

// AFTER:
@group(0) @binding(1) var<storage, read> tet_to_vert: array<u32>; // maps tet_idx -> position in uninserted array (or INVALID)
```

**split.wgsl line 24:**
```wgsl
// BEFORE:
@group(0) @binding(4) var<storage, read> insert_list: array<vec2<u32>>; // (tet_idx, vert_idx)

// AFTER:
@group(0) @binding(4) var<storage, read> insert_list: array<vec2<u32>>; // (tet_idx, position)
```

---

## 5. Verification Plan

After applying all fixes:

1. **Test insert tracking**: Verify phase1.rs correctly identifies inserted vertices
2. **Test vert_tet updates**: Verify split_points correctly updates vert_tet[vertex_id]
3. **Test concurrent splits**: Verify split.wgsl correctly resolves concurrent split neighbors
4. **Test vote indexing**: Verify votes use correct vert_tet positions
5. **Run full test suite**: All 66 tests should pass

---

## 6. Impact Assessment

### Severity: CRITICAL
- Affects correctness of insertion algorithm
- Breaks concurrent split detection
- Causes incorrect vert_tet updates
- Likely root cause of current SIGSEGV crashes

### Affected Systems
- Point insertion pipeline
- Split neighbor resolution
- vert_tet maintenance
- Uninserted list compaction

### Expected Outcome After Fix
- Concurrent splits resolve correctly
- vert_tet stays synchronized
- All tests pass
- No more SIGSEGV crashes
