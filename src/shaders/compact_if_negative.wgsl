// GPU compaction to remove negative values from an array
// Port of thrust::remove_if(IsNegative()) from ThrustWrapper.cu:212-220
//
// Two-pass_num algorithm:
// Pass 1: Count non-negative values using atomic counter
// Pass 2: Copy non-negative values to compacted output

@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;
@group(0) @binding(2) var<storage, read_write> counter: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x = input_size, y = pass_num (0 or 1)

@compute @workgroup_size(256)
fn compact_if_negative(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let input_size = params.x;
    let pass_num = params.y;

    if idx >= input_size {
        return;
    }

    let value = input[idx];

    if pass_num == 0u {
        // Pass 1: Count non-negative values
        if value >= 0 {
            atomicAdd(&counter[0], 1u);
        }
    } else {
        // Pass 2: Compact non-negative values
        if value >= 0 {
            let out_idx = atomicAdd(&counter[1], 1u);
            output[out_idx] = value;
        }
    }
}
