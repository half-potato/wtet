// Kernel: Shift infinity free indices
//
// Shifts tet indices in the infinity vertex's free list that are >= start by adding shift.
// This is used during tet array reorganization when infinity tet block is moved.
//
// Corresponds to kerShiftInfFreeIdx in KerDivision.cu line 992
//
// Dispatch: ceil(free_num / 64)

@group(0) @binding(0) var<storage, read_write> vert_free_arr: array<u32>;
@group(0) @binding(1) var<storage, read_write> free_arr: array<u32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = inf_idx, y = start, z = shift

const MEAN_VERTEX_DEGREE: u32 = 8u;

@compute @workgroup_size(64)
fn shift_inf_free_idx(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let inf_idx = params.x;
    let start = params.y;
    let shift = params.z;

    let free_num = vert_free_arr[inf_idx];
    let free_beg = inf_idx * MEAN_VERTEX_DEGREE;

    let idx = gid.x;
    if idx >= free_num {
        return;
    }

    let tet_idx = free_arr[free_beg + idx];

    // Only shift if >= start
    if tet_idx >= start {
        free_arr[free_beg + idx] = tet_idx + shift;
    }
}
