# Buffer Interpretation Bugs - Summary

## Critical Findings

After comprehensive comparison with CUDA source, I found **3 critical bugs** in how WGSL shaders interpret buffer values.

---

## Bug #1: vote.wgsl uses wrong vert_tet index

**File**: `/home/amai/gdel3d_wgpu/src/shaders/vote.wgsl`
**Line**: 43
**Severity**: CRITICAL

### Current Code:
```wgsl
let idx = gid.x;                    // idx is position in uninserted array
let vert_idx = uninserted[idx];     // vert_idx is vertex ID
let tet_idx = vert_tet[vert_idx];   // ❌ BUG: Uses vertex ID to index vert_tet
```

### CUDA Equivalent (KerPredicates.cu:166):
```cuda
for ( int idx = getCurThreadIdx(); idx < vertexArr._num; idx += getThreadNum() )
{
    const int tetIdx = vertexTetArr[ idx ];  // ✅ Uses idx (position)
    const int vert   = vertexArr._arr[ idx ];
}
```

### Root Cause:
- **CUDA**: `vertexTetArr` is a compacted array, indexed by position (0, 1, 2, ...)
- **WGPU**: `vert_tet` is indexed by **vertex ID** (per MEMORY.md 2026-02-28 lesson)

But wait - let me verify this against MEMORY.md more carefully:

### MEMORY.md says:
> **CRITICAL**: vert_tet is position-indexed (by idx in uninserted array), NOT vertex-indexed.

This contradicts point_location.wgsl which uses `vert_tet[vert_idx]`.

### Resolution Required:
We need to determine the CORRECT indexing strategy:
1. If vert_tet is position-indexed: fix point_location.wgsl line 64
2. If vert_tet is vertex-indexed: fix vote.wgsl line 43 and update MEMORY.md

**Looking at point_location.wgsl line 105**: `vert_tet[vert_idx] = tet_idx;`
This writes using vertex ID, suggesting vert_tet IS vertex-indexed.

**Looking at buffers.rs allocation**:
- Buffer size is `num_points * 4` bytes (not `num_uninserted * 4`)
- This confirms vert_tet is vertex-indexed (full array for all points)

### Conclusion:
- MEMORY.md lesson from 2026-02-28 is **WRONG**
- vert_tet IS vertex-indexed (not position-indexed)
- vote.wgsl line 43 should use `vert_tet[idx]` to match CUDA's position-based indexing

### Fix:
```wgsl
// BEFORE:
let tet_idx = vert_tet[vert_idx];

// AFTER:
// CUDA uses position-based indexing into compacted array
// WGPU has full vertex-indexed array, but we need to look up by position
// Option 1: Create separate compacted array (complex)
// Option 2: Keep using vert_idx since we update it at line 105
// Actually, vote.wgsl line 43 is CORRECT! The bug is elsewhere.
```

Wait, I'm confusing myself. Let me trace through the full data flow more carefully.

---

## Data Flow Analysis

### CUDA's Array Structure:
After `compactBothIfNegative()` (GpuDelaunay.cu:478):
- `_vertVec[i]` = vertex ID of i-th uninserted point
- `_vertTetVec[i]` = tet containing i-th uninserted point
- Both arrays have length = num_uninserted
- Arrays are compacted after each insertion round

### WGPU's Array Structure:
- `uninserted[i]` = vertex ID of i-th uninserted point (same as CUDA's `_vertVec`)
- `vert_tet[v]` = tet containing vertex v (indexed by VERTEX ID, not position!)
- `uninserted` has length = num_uninserted
- `vert_tet` has length = num_points (full array)

### Key Difference:
- **CUDA**: Both arrays compacted, indexed by position
- **WGPU**: `uninserted` compacted (position), `vert_tet` full (vertex ID)

This design choice makes sense because:
- WGPU doesn't compact buffers during execution
- Vertex IDs are stable, positions in uninserted array change

---

## Bug #2: insert_list.y interpretation

**Files**: Multiple
**Severity**: CRITICAL - Causes cascading bugs

### The Contract (per CUDA):
`insert_list[i].y` should contain the **position** in the uninserted array (NOT vertex ID).

### CUDA Evidence (KerDivision.cu:330):
```cuda
if ( winSVal == vertSVal )
    atomicMin( &tetVertArr[ tetIdx ], idx );  // Stores POSITION (idx)
```

Later (KerDivision.cu:116):
```cuda
const int insIdx = newVertVec._arr[ idx ];  // Read POSITION
const int tetIdx = makePositive( vertTetArr[ insIdx ] );
const int splitVertex = vertArr[ insIdx ];  // Convert POSITION -> VERTEX ID
```

### Current WGPU Implementation:

#### vote.wgsl (pick_winner_point) - Line 99:
```wgsl
let local_idx = unpack_vote_idx(vote);    // local_idx is POSITION
let vert_idx = uninserted2[local_idx];    // ✅ Convert POSITION -> VERTEX ID

insert_list[slot] = vec2<u32>(tet_idx, vert_idx);  // ❌ BUG: Should store local_idx!
```

**Bug**: Stores vertex ID instead of position.

#### split.wgsl - Line 123:
```wgsl
let insert = insert_list[idx];
let old_tet = insert.x;
let p = insert.y;  // Treats as vertex ID (works due to vote.wgsl bug)
```

**Currently works by accident** because vote.wgsl stores vertex ID.

#### split.wgsl - Lines 227-234:
```wgsl
let nei_split_idx = tet_to_vert[nei_tet];           // nei_split_idx is POSITION
let nei_split_vert = insert_list[nei_split_idx].y;  // ❌ Expects vertex ID, should get position
let nei_free_idx = (nei_split_vert + 1u) * MEAN_VERTEX_DEGREE - 1u;
```

**Bug**: Expects to read vertex ID from insert_list, but per contract it should be position.

---

## Bug #3: split_points.wgsl vert_tet update

**File**: `/home/amai/gdel3d_wgpu/src/shaders/split_points.wgsl`
**Line**: 151
**Severity**: CRITICAL

### Current Code:
```wgsl
let vert_idx = global_id.x;              // vert_idx is POSITION in uninserted array
let vertex = uninserted[vert_idx];       // vertex is VERTEX ID

// ... decision tree ...

vert_tet[vert_idx] = new_tet;            // ❌ BUG: Uses POSITION instead of VERTEX ID
```

### CUDA Equivalent (KerPredicates.cu:279):
```cuda
vertexTetArr[ vertIdx ] = face;  // vertIdx is POSITION in compacted array
```

### Problem:
- CUDA: `vertexTetArr` is compacted, uses position
- WGPU: `vert_tet` is full array indexed by vertex ID
- split_points.wgsl should use `vert_tet[vertex]` not `vert_tet[vert_idx]`

### Fix:
```wgsl
// BEFORE:
vert_tet[vert_idx] = new_tet;

// AFTER:
vert_tet[vertex] = new_tet;  // vertex is the actual vertex ID
```

---

## Summary of All Fixes

### Fix 1: vote.wgsl line 103
```wgsl
// BEFORE:
insert_list[slot] = vec2<u32>(tet_idx, vert_idx);

// AFTER:
insert_list[slot] = vec2<u32>(tet_idx, local_idx);  // Store POSITION not vertex ID
```

### Fix 2: split.wgsl lines 123-124
```wgsl
// BEFORE:
let p = insert.y;

// AFTER:
let position = insert.y;
let p = uninserted[position];  // Convert POSITION -> VERTEX ID
```

### Fix 3: split.wgsl line 231
```wgsl
// BEFORE:
let nei_split_vert = insert_list[nei_split_idx].y;

// AFTER:
let nei_split_position = insert_list[nei_split_idx].y;
let nei_split_vert = uninserted[nei_split_position];
```

### Fix 4: split_points.wgsl line 151
```wgsl
// BEFORE:
vert_tet[vert_idx] = new_tet;

// AFTER:
vert_tet[vertex] = new_tet;
```

### Fix 5: phase1.rs read_inserted_verts() line 358
```rust
// BEFORE:
raw.iter().map(|pair| pair[1]).collect()

// AFTER:
raw.iter()
    .map(|pair| {
        let position = pair[1] as usize;
        state.uninserted[position]
    })
    .collect()
```

### Fix 6: Update MEMORY.md
Remove incorrect lesson about "vert_tet is position-indexed".
Add correct lesson: "vert_tet is vertex-indexed (full array), unlike CUDA's compacted array."

---

## Root Cause

The confusion stems from CUDA vs WGPU architectural difference:
- **CUDA**: Compacts both `_vertVec` and `_vertTetVec` after each round
- **WGPU**: Only compacts `uninserted`, keeps `vert_tet` as full array

This is a valid design choice, but requires careful conversion between positions and vertex IDs.

---

## Expected Impact After Fixes

1. ✅ Concurrent split detection will work correctly
2. ✅ vert_tet updates will target correct vertices
3. ✅ Insert list compaction will track correct vertices
4. ✅ All 66 tests should pass
5. ✅ SIGSEGV crashes should be resolved
