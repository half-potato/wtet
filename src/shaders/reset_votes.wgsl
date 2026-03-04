// Kernel: Reset tet_vote array to NO_VOTE before each voting round.
//
// TODO: Switch to atomicMax voting with circumcenter (currently using atomicMin with centroid)

@group(0) @binding(0) var<storage, read_write> tet_vote: array<atomic<i32>>;
@group(0) @binding(1) var<uniform> params: vec4<u32>; // x = max_tets

const NO_VOTE: i32 = 0x7FFFFFFF;  // Maximum i32 for atomicMin voting

@compute @workgroup_size(64)
fn reset_votes(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    if gid.x >= params.x {
        return;
    }
    atomicStore(&tet_vote[gid.x], NO_VOTE);
}
