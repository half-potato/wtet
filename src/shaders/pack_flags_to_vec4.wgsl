// Pack u32 flags to vec4<u32> for GPU prefix sum input.
//
// Takes the compaction_flags buffer (u32 array with 0/1 values) and packs
// 4 consecutive values into vec4<u32> for b0nes164 GPU prefix sum.
//
// Bindings:
// - @binding(0): flags (read-only) - u32 flags array (0 or 1)
// - @binding(1): scan_in (write) - vec4<u32> packed values
// - @binding(2): params (uniform) - [num_elements, 0, 0, 0]

@group(0) @binding(0) var<storage, read> flags: array<u32>;
@group(0) @binding(1) var<storage, read_write> scan_in: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;

@compute @workgroup_size(256)
fn pack_flags_to_vec4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vec_idx = gid.x;
    let base_idx = vec_idx * 4u;
    let total = params.x;

    // Pack 4 consecutive flags into vec4
    var result = vec4<u32>(0u, 0u, 0u, 0u);

    if (base_idx + 0u < total) {
        result.x = flags[base_idx + 0u];
    }
    if (base_idx + 1u < total) {
        result.y = flags[base_idx + 1u];
    }
    if (base_idx + 2u < total) {
        result.z = flags[base_idx + 2u];
    }
    if (base_idx + 3u < total) {
        result.w = flags[base_idx + 3u];
    }

    scan_in[vec_idx] = result;
}
