// GPU shader to initialize tet_to_flip buffer to -1
// Replaces CPU vec allocation + PCIe transfer (20MB+ per call)
//
// Used in relocate phase to build flip trace chains.
// Much faster than CPU allocation + transfer for large datasets.

@group(0) @binding(0) var<storage, read_write> tet_to_flip: array<i32>;
@group(0) @binding(1) var<uniform> params: vec4<u32>;  // x=num_tets

@compute @workgroup_size(256)
fn init_tet_to_flip(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_tets = params.x;

    if idx >= num_tets {
        return;
    }

    tet_to_flip[idx] = -1;
}
