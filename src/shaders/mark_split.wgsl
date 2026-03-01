// Kernel: Mark which tets are splitting
//
// Before running the split kernel, we need to populate a tet_to_vert mapping
// so that split threads can detect when their neighbors are also splitting.
//
// This shader reads the insert_list and marks each tet with the vertex being inserted.

@group(0) @binding(0) var<storage, read> insert_list: array<vec2<u32>>; // (tet_idx, position)
@group(0) @binding(1) var<storage, read_write> tet_to_vert: array<u32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = num_insertions

@compute @workgroup_size(64)
fn mark_split(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_insertions = params.x;

    if idx >= num_insertions {
        return;
    }

    let insert = insert_list[idx];
    let tet_idx = insert.x;
    // Store the insertion index (idx), not the vertex!
    // This matches original line 104: tetToVert[tetIdx] stores index into vertArr/insVertVec
    tet_to_vert[tet_idx] = idx;
}
