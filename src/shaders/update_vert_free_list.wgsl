// Kernel: Allocate tet blocks for vertices being inserted in this batch.
//
// For each vertex in insert_list, allocate a block of MEAN_VERTEX_DEGREE
// tet slots. This follows CUDA's block-based allocation strategy where
// vertex v gets slots at [v * MEAN_VERTEX_DEGREE, (v+1) * MEAN_VERTEX_DEGREE).
//
// These slots are filled with consecutive tet indices starting from start_free_idx.
//
// Dispatch: ceil((num_insertions * MEAN_VERTEX_DEGREE) / 64)

@group(0) @binding(0) var<storage, read> insert_list: array<vec2<u32>>; // (tet_idx, position)
@group(0) @binding(1) var<storage, read_write> vert_free_arr: array<u32>;
@group(0) @binding(2) var<storage, read_write> free_arr: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x = num_insertions, y = start_free_idx
@group(0) @binding(4) var<storage, read> uninserted: array<u32>;

const MEAN_VERTEX_DEGREE: u32 = 8u; // From original KerCommon.h line 56

@compute @workgroup_size(64)
fn update_vert_free_list(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let num_insertions = params.x;
    let start_free_idx = params.y;
    let new_free_num = num_insertions * MEAN_VERTEX_DEGREE;

    let idx = gid.x;
    if idx >= new_free_num {
        return;
    }

    // Which insertion does this thread handle?
    let ins_idx = idx / MEAN_VERTEX_DEGREE;
    let loc_idx = idx % MEAN_VERTEX_DEGREE;

    // Get the position from insert_list, then lookup actual vertex index
    let position = insert_list[ins_idx].y;
    let vert_idx = uninserted[position];

    // Allocate tet index for this slot
    let tet_idx = start_free_idx + idx;

    // Write to the vertex's block
    free_arr[vert_idx * MEAN_VERTEX_DEGREE + loc_idx] = tet_idx;

    // Update free count for this vertex (only first thread per vertex)
    if loc_idx == 0u {
        vert_free_arr[vert_idx] = MEAN_VERTEX_DEGREE;
    }
}
