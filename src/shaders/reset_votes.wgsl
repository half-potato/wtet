// Kernel: Reset tet_vote and vert_sphere arrays before each voting round.
//
// Uses atomicMax voting with circumcenter (insphere determinant)

@group(0) @binding(0) var<storage, read_write> tet_vote: array<atomic<i32>>;
@group(0) @binding(1) var<storage, read_write> vert_sphere: array<i32>;
@group(0) @binding(2) var<storage, read_write> tet_to_vert: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x = max_tets, y = num_uninserted

const NO_VOTE: i32 = i32(0x80000000u);  // Minimum i32 for atomicMax voting
const INVALID: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(256)
fn reset_votes(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let max_tets = params.x;
    let num_uninserted = params.y;

    // Reset tet votes and tet_to_vert
    if idx < max_tets {
        atomicStore(&tet_vote[idx], NO_VOTE);
        atomicStore(&tet_to_vert[idx], INVALID);
    }

    // Reset vertex sphere values
    if idx < num_uninserted {
        vert_sphere[idx] = NO_VOTE;
    }
}
