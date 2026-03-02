// Utility kernels for updating state between iterations.

// --- kerResetVotes: Clear tet_vote array to NO_VOTE ---
@group(0) @binding(0) var<storage, read_write> tet_vote: array<atomic<i32>>;
@group(0) @binding(1) var<uniform> params: vec4<u32>; // x = max_tets

const NO_VOTE: i32 = 0x7FFFFFFF;

@compute @workgroup_size(64)
fn reset_votes(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    if gid.x >= params.x {
        return;
    }
    atomicStore(&tet_vote[gid.x], NO_VOTE);
}

// --- kerClearChanged: Clear TET_CHANGED flag on all tets ---
@group(0) @binding(0) var<storage, read_write> tet_info: array<u32>;
@group(0) @binding(1) var<uniform> params2: vec4<u32>; // x = max_tets

const TET_CHANGED: u32 = 2u;

@compute @workgroup_size(64)
fn clear_changed(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    if gid.x >= params2.x {
        return;
    }
    tet_info[gid.x] = tet_info[gid.x] & ~TET_CHANGED;
}

// --- kerRelocatePoints: After flips, update vert_tet for displaced points ---
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> tets: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> tet_opp: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> tet_info: array<u32>;
@group(0) @binding(4) var<storage, read_write> vert_tet: array<u32>;
@group(0) @binding(5) var<storage, read> uninserted: array<u32>;
@group(0) @binding(6) var<uniform> params3: vec4<u32>; // x = num_uninserted

const INVALID: u32 = 0xFFFFFFFFu;
const TET_ALIVE: u32 = 1u;

fn orient3d_simple(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> f32 {
    let ad = a - d; let bd = b - d; let cd = c - d;
    return ad.x * (bd.y * cd.z - bd.z * cd.y)
         + bd.x * (cd.y * ad.z - cd.z * ad.y)
         + cd.x * (ad.y * bd.z - ad.z * bd.y);
}

@compute @workgroup_size(64)
fn relocate_points(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    if gid.x >= params3.x {
        return;
    }

    let vert_idx = uninserted[gid.x];
    let tet_idx = vert_tet[vert_idx];

    // If current tet is still alive, we might still be inside it — keep it
    if tet_idx != INVALID && (tet_info[tet_idx] & TET_ALIVE) != 0u {
        return;
    }

    // Tet was destroyed by a flip — need to re-locate
    // Set to INVALID so point_location will start from tet 0
    vert_tet[vert_idx] = INVALID;
}
