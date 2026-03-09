// Unpack vec4<u32> scan results back to flat u32 array.
//
// Converts the vec4-packed prefix sum output from prefix_sum.wgsl back into
// a flat u32 array for compatibility with downstream compaction shaders.
//
// Bindings:
// - @binding(0): scan_out (read-only) - vec4<u32> prefix sum results
// - @binding(1): prefix_sum_data (write) - u32 flat array
// - @binding(2): params (uniform) - [total_tet_num, 0, 0, 0]

@group(0) @binding(0) var<storage, read> scan_out: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> prefix_sum_data: array<u32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;

@compute @workgroup_size(256)
fn unpack_vec4_to_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vec_idx = gid.x;
    let base_idx = vec_idx * 4u;
    let total = params.x;

    let packed = scan_out[vec_idx];

    if (base_idx + 0u < total) {
        prefix_sum_data[base_idx + 0u] = packed.x;
    }
    if (base_idx + 1u < total) {
        prefix_sum_data[base_idx + 1u] = packed.y;
    }
    if (base_idx + 2u < total) {
        prefix_sum_data[base_idx + 2u] = packed.z;
    }
    if (base_idx + 3u < total) {
        prefix_sum_data[base_idx + 3u] = packed.w;
    }
}
