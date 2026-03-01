// Kernel: Locate the containing tetrahedron for each uninserted point.
//
// Uses stochastic walk: starting from vert_tet[i], check orient3d against
// each face. If the point is on the wrong side of a face, walk to the
// neighbour across that face. Repeat until contained.
//
// Dispatch: ceil(num_uninserted / 64)

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> tets: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> tet_opp: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> tet_info: array<u32>;
@group(0) @binding(4) var<storage, read_write> vert_tet: array<u32>;
@group(0) @binding(5) var<storage, read> uninserted: array<u32>;
@group(0) @binding(6) var<uniform> params: vec4<u32>; // x = num_uninserted, y = max_walk_steps

const INVALID: u32 = 0xFFFFFFFFu;
const TET_ALIVE: u32 = 1u;
const MAX_WALK: u32 = 512u;

// Face opposite vertex i: the other 3 vertices.
// face_opp[i] gives indices into the tet's vertex array for the face opposite vertex i.
// Winding is chosen so that orient3d(face[0], face[1], face[2], vertex_i) > 0
// for a positively-oriented tet.
const FACE_OPP_0: vec3<u32> = vec3<u32>(1u, 3u, 2u); // opposite v0
const FACE_OPP_1: vec3<u32> = vec3<u32>(0u, 2u, 3u); // opposite v1
const FACE_OPP_2: vec3<u32> = vec3<u32>(0u, 3u, 1u); // opposite v2
const FACE_OPP_3: vec3<u32> = vec3<u32>(0u, 1u, 2u); // opposite v3

fn get_face_verts(face: u32) -> vec3<u32> {
    switch face {
        case 0u: { return FACE_OPP_0; }
        case 1u: { return FACE_OPP_1; }
        case 2u: { return FACE_OPP_2; }
        default: { return FACE_OPP_3; }
    }
}

fn orient3d_simple(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> f32 {
    let ad = a - d;
    let bd = b - d;
    let cd = c - d;
    return ad.x * (bd.y * cd.z - bd.z * cd.y)
         + bd.x * (cd.y * ad.z - cd.z * ad.y)
         + cd.x * (ad.y * bd.z - ad.z * bd.y);
}

@compute @workgroup_size(64)
fn locate_points(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_uninserted = params.x;
    let max_steps = select(MAX_WALK, params.y, params.y > 0u);

    if idx >= num_uninserted {
        return;
    }

    let vert_idx = uninserted[idx];
    let p = points[vert_idx].xyz;

    // Start from the last known tet - indexed by position (idx), not vertex!
    var tet_idx = vert_tet[idx];

    // If the tet is dead, start from 0
    if tet_idx == INVALID || (tet_info[tet_idx] & TET_ALIVE) == 0u {
        tet_idx = 0u;
    }

    for (var step = 0u; step < max_steps; step++) {
        let tet = tets[tet_idx];
        let opp = tet_opp[tet_idx];

        // Check each face: if point is on the wrong side, walk through
        var found = true;
        for (var f = 0u; f < 4u; f++) {
            let fv = get_face_verts(f);
            let a = points[tet[fv.x]].xyz;
            let b = points[tet[fv.y]].xyz;
            let c = points[tet[fv.z]].xyz;

            let o = orient3d_simple(a, b, c, p);

            // If orient is negative, point is on the wrong side of this face
            // (outside the tet through this face). Walk to neighbour.
            if o < 0.0 {
                let neighbour = opp[f];
                if neighbour == INVALID {
                    // Boundary — shouldn't happen with super-tet, but be safe
                    found = true;
                    break;
                }
                // Decode TetOpp: tet_idx is in upper bits (shifted left by 5)
                tet_idx = neighbour >> 5u;
                found = false;
                break;
            }
        }

        if found {
            break;
        }
    }

    // Store result - indexed by position (idx), not vertex!
    vert_tet[idx] = tet_idx;
}
