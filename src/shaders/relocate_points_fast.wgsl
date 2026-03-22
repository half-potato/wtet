// Port of kerRelocatePointsFast + kerRelocatePointsExact from gDel3D/GPU/KerPredicates.cu (lines 822-947)
// Combined into single pass: fast orient3d with error bounds + exact DD fallback
//
// Updates vert_tet for vertices whose containing tets were flipped.
// Walks the flip trace chain (tetToFlip) to find the final tet location.
//
// CUDA uses two separate kernel dispatches (fast then exact).
// We combine them: use error-bounded fast orient3d first, fall back to
// double-double exact arithmetic when uncertain, with SoS for true degeneracy.

@group(0) @binding(0) var<storage, read> uninserted: array<u32>;
@group(0) @binding(1) var<storage, read_write> vert_tet: array<i32>;
@group(0) @binding(2) var<storage, read> tet_to_flip: array<i32>;
@group(0) @binding(3) var<storage, read> flip_arr: array<vec4<i32>>; // FlipItem = 2 x vec4<i32>
@group(0) @binding(4) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> params: vec4<u32>; // x = num_uninserted, y = inf_idx

// --- Double-Double Arithmetic for Exact Predicates ---
// (Same as split_points.wgsl)

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

// Orient3d with double-double arithmetic (raw Shewchuk sign)
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

// SoS tie-breaking using vertex indices (same as split_points.wgsl)
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

// --- Orient3d with error bounds + exact fallback + infinity handling ---
// Matches CUDA's doOrient3DFast (KerPredWrapper.h:89-101) + doOrient3DSoS fallback
//
// CUDA convention (ortToOrient from CommonTypes.h:96-98):
//   det < 0 -> OrientPos (+1)
//   det > 0 -> OrientNeg (-1)
// CUDA (KerPredWrapper.h:89-101): if v0/v1/v2 is infIdx, negate det before ortToOrient.
fn orient3d_with_inf(v0: u32, v1: u32, v2: u32, v3: u32,
                     a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> i32 {
    let inf_idx = params.y;
    let has_inf = v0 >= inf_idx || v1 >= inf_idx || v2 >= inf_idx;

    // Fast path with error bounds (matching CUDA orient3dFast from KerShewchuk.h:748-794)
    let adx = a.x - d.x; let ady = a.y - d.y; let adz = a.z - d.z;
    let bdx = b.x - d.x; let bdy = b.y - d.y; let bdz = b.z - d.z;
    let cdx = c.x - d.x; let cdy = c.y - d.y; let cdz = c.z - d.z;

    let bdxcdy = bdx * cdy; let cdxbdy = cdx * bdy;
    let cdxady = cdx * ady; let adxcdy = adx * cdy;
    let adxbdy = adx * bdy; let bdxady = bdx * ady;

    var det = adz * (bdxcdy - cdxbdy)
            + bdz * (cdxady - adxcdy)
            + cdz * (adxbdy - bdxady);

    // Error bound: O3derrboundA = (7 + 56*eps)*eps for f32
    // eps = 2^-24 ~ 5.96e-8, so bound ~ 4.17e-7
    // Use slightly conservative value for safety
    let permanent = (abs(bdxcdy) + abs(cdxbdy)) * abs(adz)
                  + (abs(cdxady) + abs(adxcdy)) * abs(bdz)
                  + (abs(adxbdy) + abs(bdxady)) * abs(cdz);
    let err_bound = 7.77e-7 * permanent;

    if abs(det) > err_bound {
        // Fast result is conclusive
        if has_inf { det = -det; }
        if det < 0.0 { return 1; }  // OrientPos
        if det > 0.0 { return -1; } // OrientNeg
        return 0;
    }

    // Exact fallback using double-double arithmetic
    // (CUDA would use kerRelocatePointsExact with doOrient3DSoS)
    var exact_sign = orient3d_dd(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z, d.x, d.y, d.z);

    if exact_sign == 0 {
        // True degeneracy - use SoS tie-breaking (matches CUDA doOrient3DSoS)
        exact_sign = sos_orient3d_index(v0, v1, v2, v3);
    }

    // Negate for infinity vertices (same convention as fast path)
    if has_inf { exact_sign = -exact_sign; }

    // Convert raw Shewchuk sign to CUDA's ortToOrient convention:
    //   raw negative -> OrientPos (+1)
    //   raw positive -> OrientNeg (-1)
    if exact_sign < 0 { return 1; }  // OrientPos
    if exact_sign > 0 { return -1; } // OrientNeg
    return 0; // Should not reach here with SoS
}

fn load_flip_item(flip_idx: u32) -> array<i32, 8> {
    let v0 = flip_arr[flip_idx * 2u];
    let v1 = flip_arr[flip_idx * 2u + 1u];
    return array<i32, 8>(
        v0.x, v0.y, v0.z, v0.w,  // _v[0-3]
        v1.x,                     // _v[4]
        v1.y, v1.z, v1.w          // _t[0-2]
    );
}

@compute @workgroup_size(256)
fn relocate_points_fast(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let vert_idx = gid.x;
    let num_uninserted = params.x;

    if vert_idx >= num_uninserted {
        return;
    }

    let tet_idx_val = vert_tet[vert_idx];

    // Handle INVALID: 0xFFFFFFFF as u32 = -1 as i32
    if tet_idx_val == -1 {
        return;
    }

    // removeExactBit (clear MSB)
    let tet_idx = tet_idx_val & i32(~(1u << 31u));

    var next_idx = tet_to_flip[tet_idx];

    if next_idx == -1 {
        return; // Tet not flipped
    }

    let vertex = uninserted[vert_idx];
    let vertex_p = points[vertex].xyz;

    var flag = next_idx & 1;
    var dest_idx = next_idx >> 1;

    // Walk flip trace chain (safety limit to prevent infinite loops)
    var loop_count = 0u;
    loop {
        if flag != 1 || loop_count > 1000u {
            break;
        }
        loop_count++;

        let flip_item = load_flip_item(u32(dest_idx));

        // Determine flip type: 3-2 if _t[2] < 0, else 2-3
        let f_type = select(0u, 1u, flip_item[7] < 0); // 0 = Flip23, 1 = Flip32

        var next_loc_id: i32;

        // Pre-load flip_item vertex IDs and points
        let vid0 = u32(flip_item[0]);
        let vid1 = u32(flip_item[1]);
        let vid2 = u32(flip_item[2]);
        let vid3 = u32(flip_item[3]);
        let vid4 = u32(flip_item[4]);
        let p0 = points[vid0].xyz;
        let p1 = points[vid1].xyz;
        let p2 = points[vid2].xyz;
        let p3 = points[vid3].xyz;
        let p4 = points[vid4].xyz;

        var ord0: i32;
        if f_type == 0u {
            // Flip23: F = (0, 2, 3)
            ord0 = orient3d_with_inf(vid0, vid2, vid3, vertex, p0, p2, p3, vertex_p);
        } else {
            // Flip32: F = (0, 1, 2)
            ord0 = orient3d_with_inf(vid0, vid1, vid2, vertex, p0, p1, p2, vertex_p);
        }

        // With exact+SoS, OrientZero should not occur. But handle gracefully just in case.
        if ord0 == 0 {
            break;
        }

        if f_type == 1u {
            // Flip32
            next_loc_id = select(1, 0, ord0 == 1);
        } else {
            // Flip23
            var ord1: i32;
            if ord0 == 1 {
                next_loc_id = 0;
                // F = (0, 3, 1)
                ord1 = orient3d_with_inf(vid0, vid3, vid1, vertex, p0, p3, p1, vertex_p);
            } else {
                next_loc_id = 1;
                // F = (0, 4, 3)
                ord1 = orient3d_with_inf(vid0, vid4, vid3, vertex, p0, p4, p3, vertex_p);
            }

            if ord1 == 0 {
                break;
            }

            next_loc_id = select(next_loc_id, 2, ord1 != 1);
        }

        next_idx = flip_item[5 + next_loc_id]; // _t[next_loc_id]
        flag = next_idx & 1;
        dest_idx = next_idx >> 1;
    }

    vert_tet[vert_idx] = dest_idx;
}
