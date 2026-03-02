// Port of kerMarkTetEmpty from gDel3D/GPU/KerDivision.cu (lines 787-795)
// Called before split to set the empty flag (bit 2) on all tets
//
// CUDA code:
//   for ( int idx = getCurThreadIdx(); idx < tetInfoVec._num; idx += getThreadNum() )
//       setTetEmptyState( tetInfoVec._arr[ idx ], true );
//
// This sets bit 2 of tet_info to 1, marking the tet as "empty".
// Despite the confusing naming, this is what the CUDA code actually does.

@group(0) @binding(0) var<storage, read_write> tet_info: array<u32>;
@group(0) @binding(1) var<uniform> params: vec4<u32>; // x = num_tets

// Bit flags for tet_info (matching CUDA's bit layout)
// CUDA bit 0: Alive
// CUDA bit 1: CheckState
// CUDA bit 2: Empty
// In WGSL, we use u32 instead of char, but same bit positions
const TET_EMPTY: u32 = 4u; // Bit 2

@compute @workgroup_size(64)
fn mark_tet_empty(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_tets = params.x;

    if idx >= num_tets {
        return;
    }

    // Set bit 2 (empty flag) to 1
    // CUDA: setTetEmptyState( tetInfoVec._arr[ idx ], true );
    // This sets bit 2, leaving other bits unchanged
    tet_info[idx] = tet_info[idx] | TET_EMPTY;
}
