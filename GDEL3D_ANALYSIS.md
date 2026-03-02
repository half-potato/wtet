# gDel3D vs Current Implementation - Architectural Analysis

## Critical Differences Identified

### 1. **Split Kernel - Adjacency Updates**

**gDel3D Approach:**
- Checks `tetToVert[neighbor_tet]` to determine if neighbor has split
- If neighbor NOT split: Atomically updates neighbor's adjacency to point back
- If neighbor HAS split: Computes deterministic location of neighbor's new tet
- Uses formula: `freeArr[(neighbor_vert + 1) * MEAN_DEGREE - 1 - face_idx]`

**Current Implementation:**
- Disabled adjacency updates due to race conditions
- Results in dead tet references that break point location

**What's Needed:**
- Port the split neighbor detection logic
- Implement safe bounds checking before array access
- Handle both split and unsplit neighbor cases

### 2. **Data Structures**

**gDel3D Has:**
- `tetToVert`: Maps tet index → vertex being inserted (INT_MAX if not splitting)
- Block-based allocation per vertex with `MeanVertDegree` slots
- Deterministic allocation order (pops from end of block)

**Current Has:**
- `tet_split_map`: Equivalent to tetToVert ✓
- Block-based allocation ✓
- But: doesn't use it correctly in adjacency updates ✗

### 3. **Memory Access Patterns**

**Critical Issue in Our Code:**
The crash occurs when accessing:
- `insert_list[nei_split_idx]` - may be out of bounds
- `free_arr[nei_block_end - nei_face]` - calculation may overflow
- Need bounds validation before ANY array access

**gDel3D's Safety:**
- All array accesses are bounds-checked
- Uses shared memory for coordination within workgroups
- Atomic operations only where truly needed

### 4. **Algorithm Flow Differences**

**gDel3D Split Sequence:**
```
For each vertex to insert:
  1. Read old tet and its adjacencies
  2. Allocate 4 new tets from vertex's free block
  3. Set up internal adjacencies (between 4 new tets)
  4. For each external face:
     a. Check if neighbor has split (via tetToVert)
     b. If not split: update neighbor's opp to point to us
     c. If split: compute which of neighbor's tets to point to
  5. Write all 4 new tets
  6. Mark old tet as dead
```

**Our Current Broken Approach:**
```
  1-3. Same ✓
  4. DISABLED (causes crashes)
  5-6. Same ✓
```

## Key Missing Components

### A. Safe Neighbor Lookup
Need function that:
- Takes: neighbor_tet_idx, neighbor_face_idx
- Checks: if neighbor has split via tet_split_map
- Returns: correct tet index (old if unsplit, new if split)
- Validates: all bounds before array access

### B. Deterministic Allocation Query
The formula `(vertex + 1) * MEAN_DEGREE - 1 - face_idx` works because:
- Each vertex owns block [vertex * MEAN_DEGREE, (vertex+1) * MEAN_DEGREE)
- Allocation pops from END of block: last_idx, last_idx-1, last_idx-2, last_idx-3
- For 4-split: tet for face F is at position (block_end - F)

### C. Atomic Operations Strategy
gDel3D uses atomics ONLY for:
- Free list management (pushing/popping)
- Counter updates
- Writing to UNSPLIT neighbors (safe because they're not being modified)

Does NOT use atomics for:
- Reading from split neighbors (deterministic calculation)
- Internal adjacencies (no conflicts)

## Recommended Port Strategy

### Phase 1: Study Original Structure
1. Read through `kerSplitTetra` completely
2. Understand all edge cases they handle
3. Document every bounds check and safety measure

### Phase 2: Fix Current Implementation
1. Add `get_split_neighbor_tet()` helper function
2. Implement proper bounds checking
3. Re-enable adjacency updates with safety checks

### Phase 3: Validation
1. Test with small cases (4 points)
2. Verify no crashes
3. Check Euler characteristic on larger datasets

## Key Files to Port From

1. **KerDivision.cu** - Core splitting logic (~1200 lines)
   - `kerSplitTetra` - Main split kernel
   - Helper functions for encoding/decoding

2. **GpuDelaunay.cu** - Host-side coordination (~1400 lines)
   - `insertPoints()` - Overall insertion algorithm
   - Buffer management
   - Iteration control

3. **Data structure definitions** - Headers
   - Tet, TetOpp structures
   - Encoding schemes

## Next Steps

Given the complexity and need for careful porting, recommend:

1. **Create clean-room implementation** based on algorithm understanding
2. **Reference original for edge cases** but write fresh WGSL code
3. **Add comprehensive bounds checking** at every array access
4. **Test incrementally** with small datasets

The original code is ~4000 lines of carefully crafted CUDA with many subtle correctness details. A proper port requires systematic study and careful implementation.
