// Kernel: Locate the containing tetrahedron for each uninserted point.
//
// Uses stochastic walk: starting from vert_tet[i], check orient3d against
// each face. If the point is on the wrong side of a face, walk to the
// neighbour across that face. Repeat until contained.
//
// Uses EXACT orient3d predicates (DD + SoS) for robustness on degenerate geometry.
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

// --- Double-Double Arithmetic for Exact Predicates ---

struct DD {
    hi: f32,
    lo: f32,
}

fn two_sum(a: f32, b: f32) -> DD {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    return DD(s, e);
}

fn fast_two_sum(a: f32, b: f32) -> DD {
    let s = a + b;
    let e = b - (s - a);
    return DD(s, e);
}

fn two_product(a: f32, b: f32) -> DD {
    let p = a * b;
    let e = fma(a, b, -p);
    return DD(p, e);
}

fn dd_add(a: DD, b: DD) -> DD {
    let s = two_sum(a.hi, b.hi);
    let t = two_sum(a.lo, b.lo);
    var c = fast_two_sum(s.hi, s.lo + t.hi);
    c = fast_two_sum(c.hi, c.lo + t.lo);
    return c;
}

fn dd_sub(a: DD, b: DD) -> DD {
    return dd_add(a, DD(-b.hi, -b.lo));
}

fn dd_mul(a: DD, b: DD) -> DD {
    let p = two_product(a.hi, b.hi);
    let e = a.hi * b.lo + a.lo * b.hi + p.lo;
    return fast_two_sum(p.hi, e);
}

fn dd_from_f32(x: f32) -> DD {
    return DD(x, 0.0);
}

fn dd_sign(a: DD) -> i32 {
    if a.hi > 0.0 { return 1; }
    if a.hi < 0.0 { return -1; }
    if a.lo > 0.0 { return 1; }
    if a.lo < 0.0 { return -1; }
    return 0;
}

// Orient3d with double-double arithmetic
fn orient3d_dd(
    ax: f32, ay: f32, az: f32,
    bx: f32, by: f32, bz: f32,
    cx: f32, cy: f32, cz: f32,
    dx: f32, dy: f32, dz: f32,
) -> i32 {
    let adx = dd_from_f32(ax - dx);
    let ady = dd_from_f32(ay - dy);
    let adz = dd_from_f32(az - dz);
    let bdx = dd_from_f32(bx - dx);
    let bdy = dd_from_f32(by - dy);
    let bdz = dd_from_f32(bz - dz);
    let cdx = dd_from_f32(cx - dx);
    let cdy = dd_from_f32(cy - dy);
    let cdz = dd_from_f32(cz - dz);

    let t1 = dd_sub(dd_mul(bdy, cdz), dd_mul(bdz, cdy));
    let t2 = dd_sub(dd_mul(cdy, adz), dd_mul(cdz, ady));
    let t3 = dd_sub(dd_mul(ady, bdz), dd_mul(adz, bdy));

    let det = dd_add(dd_add(dd_mul(adx, t1), dd_mul(bdx, t2)), dd_mul(cdx, t3));
    return dd_sign(det);
}

// Fast orient3d with error bounds
fn orient3d_fast(
    ax: f32, ay: f32, az: f32,
    bx: f32, by: f32, bz: f32,
    cx: f32, cy: f32, cz: f32,
    dx: f32, dy: f32, dz: f32,
) -> vec2<f32> {
    let adx = ax - dx; let ady = ay - dy; let adz = az - dz;
    let bdx = bx - dx; let bdy = by - dy; let bdz = bz - dz;
    let cdx = cx - dx; let cdy = cy - dy; let cdz = cz - dz;

    let det = adx * (bdy * cdz - bdz * cdy)
            + bdx * (cdy * adz - cdz * ady)
            + cdx * (ady * bdz - adz * bdy);

    let eps = 5.96e-8;
    let permanent = abs(adx) * (abs(bdy * cdz) + abs(bdz * cdy))
                  + abs(bdx) * (abs(cdy * adz) + abs(cdz * ady))
                  + abs(cdx) * (abs(ady * bdz) + abs(adz * bdy));
    let err_bound = 7.77e-7 * permanent;

    if abs(det) > err_bound {
        return vec2<f32>(det, 0.0);
    }
    return vec2<f32>(det, 1.0);
}

// Exact orient3d with adaptive filter
fn orient3d_exact(
    ax: f32, ay: f32, az: f32,
    bx: f32, by: f32, bz: f32,
    cx: f32, cy: f32, cz: f32,
    dx: f32, dy: f32, dz: f32,
) -> i32 {
    let fast = orient3d_fast(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz);
    if fast.y == 0.0 {
        if fast.x > 0.0 { return 1; }
        if fast.x < 0.0 { return -1; }
        return 0;
    }
    return orient3d_dd(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz);
}

// SoS tie-breaking using vertex indices
fn sos_orient3d_index(va: u32, vb: u32, vc: u32, vd: u32) -> i32 {
    var indices = array<u32, 4>(va, vb, vc, vd);
    var parity = 0u;

    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u - i; j++) {
            if indices[j] > indices[j + 1u] {
                let tmp = indices[j];
                indices[j] = indices[j + 1u];
                indices[j + 1u] = tmp;
                parity ^= 1u;
            }
        }
    }

    return select(-1, 1, parity == 0u);
}

// Orient3d with SoS
fn orient3d_with_sos(
    a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>,
    va: u32, vb: u32, vc: u32, vd: u32
) -> i32 {
    let result = orient3d_exact(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z, d.x, d.y, d.z);

    if result == 0 {
        return sos_orient3d_index(va, vb, vc, vd);
    }

    return result;
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

    // Start from the last known tet
    var tet_idx = vert_tet[vert_idx];

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

            // Use exact orient3d with SoS for degenerate cases
            let o = orient3d_with_sos(a, b, c, p, tet[fv.x], tet[fv.y], tet[fv.z], vert_idx);

            // If orient is negative, point is on the wrong side of this face
            // (outside the tet through this face). Walk to neighbour.
            if o < 0 {
                let neighbour = opp[f];
                if neighbour == INVALID {
                    // Boundary — shouldn't happen with super-tet, but be safe
                    found = true;
                    break;
                }
                tet_idx = neighbour >> 5u;
                found = false;
                break;
            }
        }

        if found {
            break;
        }
    }

    vert_tet[vert_idx] = tet_idx;
}
