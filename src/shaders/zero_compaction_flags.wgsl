// GPU shader to zero compaction flags
// Replaces CPU vec allocation + PCIe transfer (5-10ms per iteration)
//
// Only zeros the active portion of the buffer (0..num_uninserted-1)
// instead of the entire 2M-element buffer.

@group(0) @binding(0) var<storage, read_write> flags: array<u32>;
@group(0) @binding(1) var<uniform> params: vec4<u32>;  // x=num_uninserted

@compute @workgroup_size(256)
fn zero_flags(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_uninserted = params.x;

    if idx >= num_uninserted {
        return;
    }

    flags[idx] = 0u;
}
