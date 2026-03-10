// Kernel: Make reverse mapping from scatter array
//
// Creates a reverse mapping: given old index (from scatter_arr), find new index.
// For each element in ins_vert_vec at position idx, look up its old index via
// scatter_arr, and if that old index < num, store idx as the reverse mapping.
//
// Corresponds to kerMakeReverseMap in KerDivision.cu line 816
//
// Dispatch: ceil(ins_vert_vec_size / 64)

@group(0) @binding(0) var<storage, read> ins_vert_vec: array<u32>;
@group(0) @binding(1) var<storage, read> scatter_arr: array<u32>;
@group(0) @binding(2) var<storage, read_write> rev_map_arr: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x = ins_vert_vec_size, y = num

@compute @workgroup_size(256)
fn make_reverse_map(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let ins_vert_vec_size = params.x;
    let num = params.y;

    let idx = gid.x;
    if idx >= ins_vert_vec_size {
        return;
    }

    // Get vertex index from ins_vert_vec
    let vert = ins_vert_vec[idx];

    // Look up old index via scatter_arr
    let old_idx = scatter_arr[vert];

    // If old index is valid (< num), store reverse mapping
    if old_idx < num {
        rev_map_arr[old_idx] = idx;
    }
}
