// Kernel: Reset voting arrays before each voting round
// Resets vert_sphere, tet_sphere, and tet_vert to initial values

@group(0) @binding(0) var<storage, read_write> vert_sphere: array<atomic<i32>>;
@group(0) @binding(1) var<storage, read_write> tet_sphere: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> tet_vert: array<atomic<i32>>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x = num_uninserted, y = max_tets

const INT_MAX: i32 = 0x7FFFFFFF;

@compute @workgroup_size(64)
fn reset_votes(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_uninserted = params.x;
    let max_tets = params.y;

    // Reset vert_sphere for uninserted vertices
    if idx < num_uninserted {
        atomicStore(&vert_sphere[idx], 0);
    }

    // Reset tet_sphere and tet_vert for all tets
    if idx < max_tets {
        atomicStore(&tet_sphere[idx], 0);
        atomicStore(&tet_vert[idx], INT_MAX);
    }
}
