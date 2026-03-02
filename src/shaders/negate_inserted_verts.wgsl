// Port of kerNegateInsertedVerts from gDel3D/GPU/KerDivision.cu (lines 872-886)
// Called after pick_winner to mark successfully inserted vertices
//
// CUDA code:
//   for ( int idx = getCurThreadIdx(); idx < vertTetVec._num; idx += getThreadNum() )
//   {
//       const int tetIdx = vertTetVec._arr[ idx ];
//       if ( tetToVert[ tetIdx ] == idx )
//           vertTetVec._arr[ idx ] = makeNegative( tetIdx );
//   }
//
// Logic: If a vertex's containing tet has this vertex as the winner,
// mark the vertex as inserted by negating its vert_tet value.

@group(0) @binding(0) var<storage, read_write> vert_tet: array<u32>; // position-indexed
@group(0) @binding(1) var<storage, read> tet_vert: array<i32>; // tet → winning vertex position (or INT_MAX)
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = num_uninserted

const INT_MAX: i32 = 0x7FFFFFFF;
const INVALID: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(64)
fn negate_inserted_verts(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_uninserted = params.x;

    if idx >= num_uninserted {
        return;
    }

    // CUDA line 881: const int tetIdx = vertTetVec._arr[ idx ];
    let tet_idx = vert_tet[idx];

    // Skip if already marked as inserted (INVALID in unsigned world)
    if tet_idx == INVALID {
        return;
    }

    // CUDA line 883: if ( tetToVert[ tetIdx ] == idx )
    // Check if this tet's winner is this vertex
    let winner_pos = tet_vert[tet_idx];

    if winner_pos == i32(idx) {
        // CUDA line 884: vertTetVec._arr[ idx ] = makeNegative( tetIdx );
        // In CUDA, makeNegative negates the signed integer.
        // In our unsigned representation, we use INVALID (0xFFFFFFFF) to mark inserted vertices.
        vert_tet[idx] = INVALID;
    }
}
