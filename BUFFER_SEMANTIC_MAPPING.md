# Buffer Semantic Mapping: CUDA ↔ WGSL

**Purpose**: This document maps all critical buffers between CUDA gDel3D and WGSL gDel3D implementations, documenting what each buffer semantically stores, how it's indexed, and any discovered mismatches.

**Created**: 2026-03-03
**Status**: Reference document for debugging

---

## 1. Point Storage

### CUDA: `_pointVec` (Point3DVec)
- **Type**: `DevVector<Point3>` → array of `Point3` structs (3 × RealType)
- **Size**: `_pointNum` (includes N real points + 1 infinity point at end)
- **Indexed by**: **Vertex ID** (0..N-1 = real points, N = infinity)
- **Stores**: **Geometric coordinates** of all points in the triangulation
- **Notes**:
  - Sorted by Morton code if `!_params.noSorting` (line 672-678)
  - Infinity point added at `_infIdx = _pointNum - 1` (line 301)

### WGSL: `points` (GpuBuffers)
- **Type**: `Buffer` → array of `vec4<f32>` (w=1.0 for real points, w=0.0 for infinity)
- **Size**: `num_points + 4` (N points + 4 super-tet vertices)
- **Indexed by**: **Vertex ID** (0..N+3)
- **Stores**: **Geometric coordinates** of all points
- **Notes**:
  - Uses vec4 for GPU alignment (w component unused)
  - Includes 4 super-tet vertices at indices N..N+3

**MATCH**: ✅ Both store coordinates indexed by vertex ID

---

## 2. Uninserted Vertex Lists

### CUDA: `_vertVec` (IntDVec)
- **Type**: `DevVector<int>` → array of `int`
- **Size**: Dynamic, shrinks as points are inserted
- **Indexed by**: **Position** (0..num_uninserted-1)
- **Stores**: **Vertex IDs** of points not yet inserted
- **Notes**:
  - Line 359: `thrust::sequence(_vertVec.begin(), _vertVec.end())` → initially [0, 1, 2, ..., N-1]
  - Line 908: `compactBothIfNegative(_vertTetVec, _vertVec)` → removes inserted vertices
  - Line 839: `thrust::gather(newVertVec, _vertVec, realInsVertVec)` → converts positions to vertex IDs

### WGSL: `uninserted` (GpuBuffers)
- **Type**: `Buffer` → array of `u32`
- **Size**: `num_points` (max capacity)
- **Indexed by**: **Position** (0..num_uninserted-1)
- **Stores**: **Vertex IDs** of uninserted points
- **Notes**:
  - Dynamically sized via `state.uninserted: Vec<u32>` in Rust

**MATCH**: ✅ Both store vertex IDs indexed by position

---

## 3. Vertex → Tet Mapping

### CUDA: `_vertTetVec` (IntDVec)
- **Type**: `DevVector<int>` → array of `int`
- **Size**: `_pointNum` (all vertices including super-tet)
- **Indexed by**: **Position in uninserted array** (NOT vertex ID!)
- **Stores**: **Tet index** containing that uninserted point
- **Notes**:
  - Line 343: `_vertTetVec.assign(_pointNum, 0)` → init to tet 0
  - Line 217: `vertexTetArr[vertIdx]` where `vertIdx` is position in `vertexArr`
  - Line 823: `kerNegateInsertedVerts` negates entries to mark insertion
  - **CRITICAL**: Index is position, NOT vertex ID!

### WGSL: `vert_tet` (GpuBuffers)
- **Type**: `Buffer` → array of `u32`
- **Size**: `(num_points + 4) * 4` bytes
- **Indexed by**: **Position in uninserted array**
- **Stores**: **Tet index** containing that uninserted point
- **Notes**:
  - MEMORY fix (2026-02-28): Changed from vertex-indexed to position-indexed

**MATCH**: ✅ Both indexed by position (after 2026-02-28 fix)

---

## 4. Tet → Vertex Mapping (Split Detection)

### CUDA: `tetToVert` (IntDVec, temporary)
- **Type**: `DevVector<int>` → array of `int`
- **Size**: `max_tets`
- **Indexed by**: **Tet index**
- **Stores**: **Position** in uninserted array of vertex that will split this tet (`INT_MAX` if not splitting)
- **Notes**:
  - Line 803: `tetToVert.assign(tetNum, INT_MAX)` → init to INT_MAX
  - Line 806-811: `kerPickWinnerPoint` writes position via `atomicMin(&tetVertArr[tetIdx], idx)`
  - Line 146: `tetToVert[neiTetIdx]` used to detect concurrent split in `kerSplitTetra`
  - Line 226: `splitVertIdx = tetToVert[tetIdx]` in `kerSplitPointsFast`
  - Line 237: `vertex = vertexArr._arr[vertIdx]` → convert position to vertex ID
  - **CRITICAL**: Stores **position** (idx in vertexArr), NOT vertex ID!

### WGSL: `tet_to_vert` (GpuBuffers)
- **Type**: `Buffer` → array of `u32`
- **Size**: `max_tets * 4` bytes
- **Indexed by**: **Tet index**
- **Stores**: **Position** in uninserted array (INVALID if not splitting)
- **Notes**:
  - Initialized to INVALID (0xFFFFFFFF)
  - Used in split.wgsl for concurrent split detection

**MATCH**: ✅ Both store position indexed by tet (verified 2026-03-03)

---

## 5. Inserted Vertices Accumulator

### CUDA: `_insVertVec` (IntDVec)
- **Type**: `DevVector<int>` → array of `int`
- **Size**: Grows each iteration (line 469: `oldInsNum → newInsNum`)
- **Indexed by**: **Insertion index** (cumulative across all iterations)
- **Stores**: **Vertex IDs** of all inserted vertices (in insertion order)
- **Notes**:
  - Line 475: `thrust::copy(newVertVec, _insVertVec.begin() + oldInsNum)` → append new insertions
  - Line 183: Used to map tet block → vertex: `vertIdx = insVertVec._arr[blkIdx]`
  - Line 497: Same pattern in `kerFlip`
  - **Purpose**: Maps tet block index to vertex ID for block-based allocation

### WGSL: `ins_vert_vec` (GpuBuffers)
- **Type**: `Buffer` → array of `u32`
- **Size**: `max_tets * 4` bytes
- **Indexed by**: **Insertion index**
- **Stores**: **Vertex IDs** of inserted vertices
- **Notes**:
  - Used in expand_tetra_list / block allocation kernels

**MATCH**: ✅ Both store vertex IDs indexed by insertion order

---

## 6. Free Tet Management

### CUDA: `_freeVec` / `freeArr` (IntDVec)
- **Type**: `DevVector<int>` → array of `int`
- **Size**: `max_tets`
- **Indexed by**: **Per-vertex free list index** (block-based)
  - Vertex V's free slots: `freeArr[V * MeanVertDegree + 0 .. V * MeanVertDegree + 7]`
- **Stores**: **Tet indices** available for allocation
- **Notes**:
  - Line 120-121 (kerSplitTetra): `newIdx = (splitVertex + 1) * MeanVertDegree - 1` → last slot of block
  - Line 121: `newTetIdx[4] = {freeArr[newIdx], freeArr[newIdx-1], ...}` → pop 4 slots
  - Line 591-596 (kerUpdateVertFreeList): Initialize blocks for new vertices
  - Line 1143: `freeArr[vertIdx * MeanVertDegree + locIdx] = startFreeIdx + idx`

### WGSL: `free_arr` (GpuBuffers)
- **Type**: `Buffer` → array of `u32`
- **Size**: `max_tets * 4` bytes
- **Indexed by**: **V * MEAN_VERTEX_DEGREE + slot**
- **Stores**: **Tet indices** available per vertex
- **Notes**:
  - buffers.rs:174-178: Pre-initialized with block layout
  - Each vertex gets 8 consecutive slots

**MATCH**: ✅ Both use block-based allocation indexed by `vertex * 8 + slot`

---

## 7. Per-Vertex Free Counts

### CUDA: `_vertFreeVec` / `vertFreeArr` (IntDVec)
- **Type**: `DevVector<int>` → array of `int`
- **Size**: `_pointNum` (all vertices)
- **Indexed by**: **Vertex ID**
- **Stores**: **Count** of free tet slots available for this vertex
- **Notes**:
  - Line 648: `_vertFreeVec.assign(_pointNum, 0)` → init to 0
  - Line 307: `_vertFreeVec[_infIdx] = 0` → infinity block starts empty
  - Line 124: `vertFreeArr[splitVertex] -= 4` → consume 4 slots on split
  - Line 982: `vertFreeArr[vert] = MeanVertDegree` → new vertices get 8 slots
  - Line 1147: Same pattern in kerUpdateVertFreeList

### WGSL: `vert_free_arr` (GpuBuffers)
- **Type**: `Buffer` → array of `u32`
- **Size**: `(num_points + 4) * 4` bytes
- **Indexed by**: **Vertex ID**
- **Stores**: **Count** of free slots for each vertex
- **Notes**:
  - buffers.rs:187: Init to MEAN_VERTEX_DEGREE (8) for all vertices

**MATCH**: ✅ Both store free counts indexed by vertex ID

---

## 8. Insert List (Vote Winners)

### CUDA: `newVertVec` (temporary in `splitTetra()`)
- **Type**: `IntDVec&` (alias to `_vertSphereVec`)
- **Size**: `_insNum` (number of insertions this round)
- **Indexed by**: **Insert index** (0.._insNum-1)
- **Stores**: **Positions** in `_vertVec` of vertices to insert
- **Notes**:
  - Line 828-831: `newVertVec` populated by `thrust_copyIf_Insertable(_vertTetVec, newVertVec)`
  - Line 116 (kerSplitTetra): `insIdx = newVertVec._arr[idx]` → get position
  - Line 119: `splitVertex = vertArr[insIdx]` → convert position to vertex ID
  - **CRITICAL**: Stores **positions**, not vertex IDs!

### WGSL: `insert_list` (GpuBuffers)
- **Type**: `Buffer` → array of `vec2<u32>`
- **Size**: `num_points * 8` bytes
- **Indexed by**: **Insert index**
- **Stores**: `(tet_idx, position)` pairs (NOT vertex IDs!)
- **Notes**:
  - `.y` field should be position in uninserted array

**MATCH**: ✅ Both store positions (after understanding CUDA semantics)

---

## 9. Tetrahedra

### CUDA: `_tetVec` (TetDVec)
- **Type**: `DevVector<Tet>` → array of `Tet` structs (4 × int)
- **Size**: Dynamic (`max_tets`)
- **Indexed by**: **Tet index**
- **Stores**: **4 vertex IDs** forming the tetrahedron
- **Notes**:
  - Line 128 (kerSplitTetra): `tet = loadTet(tetArr, tetIdx)`
  - Line 167-171: `newTet = {tet._v[...], ..., splitVertex}`
  - Vertex indices are **vertex IDs** (not positions!)

### WGSL: `tets` (GpuBuffers)
- **Type**: `Buffer` → array of `vec4<u32>`
- **Size**: `max_tets * 16` bytes
- **Indexed by**: **Tet index**
- **Stores**: **4 vertex IDs**

**MATCH**: ✅ Both store vertex IDs indexed by tet index

---

## 10. Adjacency (TetOpp)

### CUDA: `_oppVec` (TetOppDVec)
- **Type**: `DevVector<TetOpp>` → array of `TetOpp` structs (4 × int)
- **Size**: `max_tets`
- **Indexed by**: **Tet index**
- **Stores**: **Packed (tet_idx, face_vi)** for each of 4 faces
- **Encoding** (CommonTypes.h:248-265):
  ```cpp
  // Bits 0-1: vi (vertex index / face)
  // Bit 2: internal flag
  // Bit 3: special flag
  // Bit 4: sphere_fail flag
  // Bits 5-31: tet_idx
  int getOppValTet(int val) { return (val >> 5); }
  int getOppValVi(int val) { return (val & 3); }
  int makeOppVal(int tetIdx, int oppTetVi) { return (tetIdx << 5) | oppTetVi; }
  ```
- **Notes**:
  - **5-bit shift**, NOT 2-bit!
  - Flags occupy bits 2-4

### WGSL: `tet_opp` (GpuBuffers)
- **Type**: `Buffer` → array of `vec4<u32>`
- **Size**: `max_tets * 16` bytes
- **Indexed by**: **Tet index**
- **Stores**: **Packed (tet_idx, face_vi)** for each of 4 faces
- **Encoding**: Should match CUDA (5-bit shift)
- **BUG**: split.wgsl still uses 2-bit encoding (SPLIT_ANALYSIS.md)

**MISMATCH**: ⚠️ split.wgsl uses `<< 2u` instead of `<< 5u` (needs fix)

---

## 11. Summary Table

| Buffer | CUDA Name | WGSL Name | Indexed By | Stores | Notes |
|--------|-----------|-----------|------------|--------|-------|
| **Points** | `_pointVec` | `points` | Vertex ID | Coordinates (xyz) | ✅ Match |
| **Uninserted list** | `_vertVec` | `uninserted` | Position | Vertex IDs | ✅ Match |
| **Vert→Tet** | `_vertTetVec` | `vert_tet` | Position | Tet index | ✅ Match (after fix) |
| **Tet→Vert** | `tetToVert` | `tet_to_vert` | Tet index | **Position** (not vertex!) | ✅ Match |
| **Inserted accum** | `_insVertVec` | `ins_vert_vec` | Insertion order | Vertex IDs | ✅ Match |
| **Free tet slots** | `freeArr` | `free_arr` | `vert*8 + slot` | Tet indices | ✅ Match |
| **Free counts** | `vertFreeArr` | `vert_free_arr` | Vertex ID | Count | ✅ Match |
| **Insert list** | `newVertVec` | `insert_list` | Insert index | **Positions** | ✅ Match |
| **Tets** | `_tetVec` | `tets` | Tet index | 4 vertex IDs | ✅ Match |
| **Adjacency** | `_oppVec` | `tet_opp` | Tet index | Packed (tet, face) | ⚠️ split.wgsl encoding bug |

---

## 12. Critical Insights

### 12.1 Position vs Vertex ID Distinction

The most subtle bug in our port was **confusing positions with vertex IDs**:

- **Position**: Index into `uninserted` / `_vertVec` array (0..num_uninserted-1)
- **Vertex ID**: Index into `points` / `_pointVec` array (0..N+3)

**Where this matters**:
1. `vert_tet[position]` → indexed by position, stores tet
2. `tet_to_vert[tet_idx]` → stores **position** (not vertex ID!)
3. `insert_list[i].y` → stores **position** (not vertex ID!)
4. `uninserted[position]` → converts position → vertex ID

**Example**:
```
uninserted = [7, 23, 5, 12]  // vertex IDs
vert_tet = [3, 0, 1, 2]      // indexed by position
// Position 0 (vertex 7) is in tet 3
// Position 1 (vertex 23) is in tet 0
```

### 12.2 TetOpp Encoding: 5-bit NOT 2-bit

CUDA uses **5-bit** shift for tet_idx, leaving bits 0-4 for flags:
```
Bits 0-1: vi
Bit 2:    internal flag
Bit 3:    special flag
Bit 4:    sphere_fail flag
Bits 5+:  tet_idx
```

**WGSL bug**: split.wgsl:62-72 uses `<< 2u` instead of `<< 5u`, causing all split adjacency to be wrong by factor of 8!

### 12.3 Block-Based Allocation

CUDA allocates tet slots in **vertex-based blocks**:
- Each vertex V owns tets `[V*8, V*8+1, ..., V*8+7]`
- `freeArr[V*8 + slot]` stores available tet indices for vertex V
- `vertFreeArr[V]` counts how many slots V has free (0..8)

This is **pre-allocated** in WGPU, so no dynamic expansion needed (unlike CUDA).

---

## 13. References

**CUDA Source**:
- `gDel3D/GDelFlipping/src/gDel3D/GpuDelaunay.cu` (host code)
- `gDel3D/GDelFlipping/src/gDel3D/GPU/KerDivision.cu` (split/flip kernels)
- `gDel3D/GDelFlipping/src/gDel3D/GPU/KerPredicates.cu` (split_points)
- `gDel3D/GDelFlipping/src/gDel3D/CommonTypes.h` (TetOpp encoding)

**WGSL Port**:
- `src/gpu/buffers.rs` (buffer allocation)
- `src/shaders/*.wgsl` (kernels)
- `MEMORY.md` (project memory)
- `SPLIT_ANALYSIS.md` (split.wgsl encoding bug)

---

**Last Updated**: 2026-03-03
