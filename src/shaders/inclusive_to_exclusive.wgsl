// Convert inclusive prefix sum to exclusive prefix sum.
//
// Inclusive: result[i] = sum(input[0..=i])  (includes element i)
// Exclusive: result[i] = sum(input[0..i])   (excludes element i)
//
// Conversion: exclusive[i] = inclusive[i] - input[i]
//
// Bindings:
// - @binding(0): flags (read-only) - original input values (0 or 1)
// - @binding(1): prefix_sum_data (read_write) - inclusive prefix sum, will be overwritten with exclusive
// - @binding(2): params (uniform) - [num_elements, 0, 0, 0]

@group(0) @binding(0) var<storage, read> flags: array<u32>;
@group(0) @binding(1) var<storage, read_write> prefix_sum_data: array<u32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;

@compute @workgroup_size(256)
fn inclusive_to_exclusive(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.x;

    if idx >= total {
        return;
    }

    // Convert: exclusive = inclusive - input
    let inclusive_value = prefix_sum_data[idx];
    let input_value = flags[idx];
    prefix_sum_data[idx] = inclusive_value - input_value;
}
