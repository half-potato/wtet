// Kernel: Shift tet indices in an array
//
// Shifts tet indices >= start by adding shift (positive indices).
// Also shifts negative indices <= -start by subtracting shift.
// Negative indices are encoded as -(v + 2) to escape -1 (special value).
//
// Corresponds to kerShiftTetIdx in KerDivision.cu line 1105
//
// Dispatch: ceil(array_size / 64)

@group(0) @binding(0) var<storage, read_write> idx_arr: array<i32>;
@group(0) @binding(1) var<uniform> params: vec4<u32>; // x = array_size, y = start, z = shift

// Helper functions for negative encoding
fn make_positive(v: i32) -> i32 {
    // Decode negative: -(v + 2)
    return -(v + 2);
}

fn make_negative(v: i32) -> i32 {
    // Encode negative: -(v + 2)
    return -(v + 2);
}

@compute @workgroup_size(64)
fn shift_tet_idx(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let array_size = params.x;
    let start = i32(params.y);
    let shift = i32(params.z);

    let idx = gid.x;
    if idx >= array_size {
        return;
    }

    let neg_start = -start;
    let old_idx = idx_arr[idx];

    // Shift positive indices >= start
    if old_idx >= start {
        idx_arr[idx] = old_idx + shift;
    }
    // Shift negative indices <= neg_start
    else if old_idx <= neg_start {
        idx_arr[idx] = old_idx - shift;
    }
}
