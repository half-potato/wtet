// Transform tet_info flags to binary 0/1 values for prefix sum input.
//
// Extracts TET_ALIVE bit (bit 0) from each tet's flags and packs 4 consecutive
// values into vec4<u32> for efficient GPU prefix sum processing.
//
// Bindings:
// - @binding(0): tet_info (read-only) - u32 flags array
// - @binding(1): scan_in (write) - vec4<u32> packed binary values
// - @binding(2): params (uniform) - [total_tet_num, 0, 0, 0]

@group(0) @binding(0) var<storage, read> tet_info: array<u32>;
@group(0) @binding(1) var<storage, read_write> scan_in: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;

const TET_ALIVE: u32 = 1u;

@compute @workgroup_size(256)
fn transform_tet_alive(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vec_idx = gid.x;
    let base_idx = vec_idx * 4u;
    let total = params.x;

    // Pack 4 consecutive flags into vec4
    var result = vec4<u32>(0u, 0u, 0u, 0u);

    if (base_idx + 0u < total) {
        result.x = tet_info[base_idx + 0u] & TET_ALIVE;
    }
    if (base_idx + 1u < total) {
        result.y = tet_info[base_idx + 1u] & TET_ALIVE;
    }
    if (base_idx + 2u < total) {
        result.z = tet_info[base_idx + 2u] & TET_ALIVE;
    }
    if (base_idx + 3u < total) {
        result.w = tet_info[base_idx + 3u] & TET_ALIVE;
    }

    scan_in[vec_idx] = result;
}
