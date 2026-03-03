// Kernel: Update tet indices using order remapping
//
// Updates tet indices in an array using a remapping table (order_arr).
// Handles both positive and negative indices (negative encoded as -(v + 2)).
// Supports block-based allocation where tets are grouped by insertion index.
//
// Corresponds to kerUpdateTetIdx in KerDivision.cu line 1050
//
// Dispatch: ceil(array_size / 64)

@group(0) @binding(0) var<storage, read_write> idx_arr: array<i32>;
@group(0) @binding(1) var<storage, read> order_arr: array<u32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = array_size, y = old_inf_block_idx, z = new_inf_block_idx

const MEAN_VERTEX_DEGREE: u32 = 8u;

// Helper functions for negative encoding
fn make_positive(v: i32) -> i32 {
    return -(v + 2);
}

fn make_negative(v: i32) -> i32 {
    return -(v + 2);
}

@compute @workgroup_size(64)
fn update_tet_idx(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let array_size = params.x;
    let old_inf_block_idx = params.y;
    let new_inf_block_idx = params.z;

    let idx = gid.x;
    if idx >= array_size {
        return;
    }

    let tet_idx = idx_arr[idx];

    // Convert to positive for processing
    var pos_tet_idx: u32;
    let is_negative = tet_idx < 0;
    if is_negative {
        pos_tet_idx = u32(make_positive(tet_idx));
    } else {
        pos_tet_idx = u32(tet_idx);
    }

    // Remap using order array or shift for infinity block
    var new_pos_tet_idx: u32;
    if pos_tet_idx < old_inf_block_idx {
        // Block-based: extract insertion index and local index
        let ins_idx = pos_tet_idx / MEAN_VERTEX_DEGREE;
        let loc_idx = pos_tet_idx % MEAN_VERTEX_DEGREE;

        // Remap using order array
        new_pos_tet_idx = order_arr[ins_idx] * MEAN_VERTEX_DEGREE + loc_idx;
    } else {
        // Infinity block: just shift
        new_pos_tet_idx = pos_tet_idx - old_inf_block_idx + new_inf_block_idx;
    }

    // Convert back to negative if needed
    if is_negative {
        idx_arr[idx] = make_negative(i32(new_pos_tet_idx));
    } else {
        idx_arr[idx] = i32(new_pos_tet_idx);
    }
}
