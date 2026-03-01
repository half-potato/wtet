// Port of kerPickWinnerPoint from gDel3D/GPU/KerDivision.cu (lines 310-334)
// Determines which uninserted vertex wins for each tet

@group(0) @binding(0) var<storage, read> uninserted: array<u32>; // vertexArr._arr in original
@group(0) @binding(1) var<storage, read> vert_tet: array<u32>; // vertexTetArr - indexed by idx!
@group(0) @binding(2) var<storage, read_write> vert_sphere: array<atomic<i32>>; // vertSphereArr - indexed by idx!
@group(0) @binding(3) var<storage, read_write> tet_sphere: array<atomic<i32>>; // tetSphereArr in original
@group(0) @binding(4) var<storage, read_write> tet_vert: array<atomic<i32>>; // tetVertArr in original
@group(0) @binding(5) var<uniform> params: vec4<u32>; // x = num_uninserted

const INVALID: i32 = 0x7FFFFFFF;

// Direct port of kerPickWinnerPoint (lines 310-334)
@compute @workgroup_size(64)
fn pick_winner_point(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_uninserted = params.x;

    // Line 321: for ( int idx = getCurThreadIdx(); idx < vertexArr._num; idx += getThreadNum() )
    if idx >= num_uninserted {
        return;
    }

    // Lines 323-325: EXACT MATCH to original - all indexed by idx!
    let vert_s_val = atomicLoad(&vert_sphere[idx]);   // Line 323: vertSphereArr[idx]
    let tet_idx = vert_tet[idx];                       // Line 324: vertexTetArr[idx]
    let win_s_val = atomicLoad(&tet_sphere[tet_idx]); // Line 325: tetSphereArr[tetIdx]

    // Lines 329-330: Check if vertex is winner
    if win_s_val == vert_s_val {
        atomicMin(&tet_vert[tet_idx], i32(idx));  // Store idx, not vertex!
    }
}
