// Port of kerCheckDelaunayExact - uses exact predicates (DD + SoS)
// Based on check_delaunay_fast.wgsl but with exact arithmetic for degenerate cases
//
// For each active tet, checks all 4 faces:
// 1. Loads neighbor opposite vertices
// 2. Checks for 3-2 flip configuration (3 tets share an edge)
// 3. Does exact insphere test to detect violations
// 4. Checks exact orient3d for 2-3 flip feasibility
// 5. Votes for the flip by writing to vote arrays

@group(0) @binding(0) var<storage, read_write> act_tet_vec: array<i32>;
@group(0) @binding(1) var<storage, read> tets: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> tet_opp: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> tet_info: array<u32>;
@group(0) @binding(4) var<storage, read_write> tet_vote_arr: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> vote_arr: array<i32>;
@group(0) @binding(6) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(8) var<uniform> params: vec4<u32>; // x = act_tet_num, y = vote_offset

const TET_ALIVE: u32 = 1u;
const TET_CHANGED: u32 = 2u;
const OPP_SPHERE_FAIL: u32 = 16u;  // Bit 4 of TetOpp encoding

const COUNTER_FLIP: u32 = 2u;

// TetViAsSeenFrom[vi] gives the 3 other vertices in counter-clockwise order when seen from vi
const TET_VI_AS_SEEN_FROM: array<vec3<u32>, 4> = array<vec3<u32>, 4>(
    vec3<u32>(1u, 3u, 2u), // seen from vertex 0
    vec3<u32>(0u, 2u, 3u), // seen from vertex 1
    vec3<u32>(0u, 3u, 1u), // seen from vertex 2
    vec3<u32>(0u, 1u, 2u), // seen from vertex 3
);

// CRITICAL FIX: Helper function to avoid variable array indexing (causes SIGSEGV)
// Cannot use TET_VI_AS_SEEN_FROM[variable] - must use explicit branches
fn get_tet_vi_as_seen_from(vi: u32) -> vec3<u32> {
    if vi == 0u { return vec3<u32>(1u, 3u, 2u); }
    else if vi == 1u { return vec3<u32>(0u, 2u, 3u); }
    else if vi == 2u { return vec3<u32>(0u, 3u, 1u); }
    else { return vec3<u32>(0u, 1u, 2u); }
}

fn decode_opp_tet(opp: u32) -> u32 {
    return opp >> 5u;
}

fn decode_opp_vi(opp: u32) -> u32 {
    return opp & 3u;
}

fn is_tet_alive(info: u32) -> bool {
    return (info & TET_ALIVE) != 0u;
}

fn is_tet_changed(info: u32) -> bool {
    return (info & TET_CHANGED) != 0u;
}

fn is_opp_internal(opp: u32) -> bool {
    let opp_tet = decode_opp_tet(opp);
    return opp_tet == 0xFFFFFFFFu;
}

fn make_negative(val: i32) -> i32 {
    return -val - 1;
}

fn make_positive(val: i32) -> i32 {
    if val >= 0 {
        return val;
    }
    return -val - 1;
}

fn make_flip(bot_vi: u32, bot_cor_ord_vi: u32) -> u32 {
    return (bot_cor_ord_vi << 2u) | bot_vi;
}

fn make_vote_val(bot_ti: u32, flip_info: u32) -> i32 {
    return i32((bot_ti << 8u) | flip_info);
}

fn vote_for_flip23(vote_offset: u32, bot_ti: u32, top_ti: u32) {
    let vote_val = i32(bot_ti);
    atomicMin(&tet_vote_arr[vote_offset + bot_ti], vote_val);
    atomicMin(&tet_vote_arr[vote_offset + top_ti], vote_val);
}

fn vote_for_flip32(vote_offset: u32, bot_ti: u32, top_ti: u32, bot_opp_ti: u32) {
    let vote_val = i32(bot_ti);
    atomicMin(&tet_vote_arr[vote_offset + bot_ti], vote_val);
    atomicMin(&tet_vote_arr[vote_offset + top_ti], vote_val);
    atomicMin(&tet_vote_arr[vote_offset + bot_opp_ti], vote_val);
}

// --- SoS (Simulation of Simplicity) tie-breaking ---

// Simple index-based SoS for insphere (5 vertices)
// Uses bubble sort to compute permutation parity
fn sos_insphere_index(va: u32, vb: u32, vc: u32, vd: u32, ve: u32) -> i32 {
    var indices = array<u32, 5>(va, vb, vc, vd, ve);
    var parity = 0u;

    // Bubble sort to count swaps
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u - i; j++) {
            if indices[j] > indices[j + 1u] {
                // Swap
                let tmp = indices[j];
                indices[j] = indices[j + 1u];
                indices[j + 1u] = tmp;
                parity ^= 1u;
            }
        }
    }

    // Even parity = +1, odd parity = -1
    return select(-1, 1, parity == 0u);
}

// Simple index-based SoS for orient3d (4 vertices)
fn sos_orient3d_index(va: u32, vb: u32, vc: u32, vd: u32) -> i32 {
    var indices = array<u32, 4>(va, vb, vc, vd);
    var parity = 0u;

    // Bubble sort to count swaps
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

// --- Include predicates.wgsl functions (DD arithmetic + exact predicates) ---
// NOTE: In actual implementation, predicates.wgsl would be included here.
// For now, we declare the functions we need:

// Double-double structure
struct DD {
    hi: f32,
    lo: f32,
}

// Core DD operations
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

// Exact orient3d using double-double arithmetic
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

// Exact insphere using double-double arithmetic
fn insphere_dd(
    ax: f32, ay: f32, az: f32,
    bx: f32, by: f32, bz: f32,
    cx: f32, cy: f32, cz: f32,
    dx: f32, dy: f32, dz: f32,
    ex: f32, ey: f32, ez: f32,
) -> i32 {
    let aex = dd_from_f32(ax - ex);
    let aey = dd_from_f32(ay - ey);
    let aez = dd_from_f32(az - ez);
    let bex = dd_from_f32(bx - ex);
    let bey = dd_from_f32(by - ey);
    let bez = dd_from_f32(bz - ez);
    let cex = dd_from_f32(cx - ex);
    let cey = dd_from_f32(cy - ey);
    let cez = dd_from_f32(cz - ez);
    let dex = dd_from_f32(dx - ex);
    let dey = dd_from_f32(dy - ey);
    let dez = dd_from_f32(dz - ez);

    let ab = dd_sub(dd_mul(aex, bey), dd_mul(bex, aey));
    let bc = dd_sub(dd_mul(bex, cey), dd_mul(cex, bey));
    let cd = dd_sub(dd_mul(cex, dey), dd_mul(dex, cey));
    let da = dd_sub(dd_mul(dex, aey), dd_mul(aex, dey));
    let ac = dd_sub(dd_mul(aex, cey), dd_mul(cex, aey));
    let bd = dd_sub(dd_mul(bex, dey), dd_mul(dex, bey));

    let abc = dd_add(dd_sub(dd_mul(dd_from_f32(aez.hi), bc), dd_mul(dd_from_f32(bez.hi), ac)), dd_mul(dd_from_f32(cez.hi), ab));
    let bcd = dd_add(dd_sub(dd_mul(dd_from_f32(bez.hi), cd), dd_mul(dd_from_f32(cez.hi), bd)), dd_mul(dd_from_f32(dez.hi), bc));
    let cda = dd_add(dd_add(dd_mul(dd_from_f32(cez.hi), da), dd_mul(dd_from_f32(dez.hi), ac)), dd_mul(dd_from_f32(aez.hi), cd));
    let dab = dd_add(dd_add(dd_mul(dd_from_f32(dez.hi), ab), dd_mul(dd_from_f32(aez.hi), bd)), dd_mul(dd_from_f32(bez.hi), da));

    let alift = dd_add(dd_add(dd_mul(aex, aex), dd_mul(aey, aey)), dd_mul(aez, aez));
    let blift = dd_add(dd_add(dd_mul(bex, bex), dd_mul(bey, bey)), dd_mul(bez, bez));
    let clift = dd_add(dd_add(dd_mul(cex, cex), dd_mul(cey, cey)), dd_mul(cez, cez));
    let dlift = dd_add(dd_add(dd_mul(dex, dex), dd_mul(dey, dey)), dd_mul(dez, dez));

    let det = dd_add(
        dd_sub(dd_mul(dlift, abc), dd_mul(clift, dab)),
        dd_sub(dd_mul(blift, cda), dd_mul(alift, bcd))
    );

    return dd_sign(det);
}

fn insphere_fast(
    ax: f32, ay: f32, az: f32,
    bx: f32, by: f32, bz: f32,
    cx: f32, cy: f32, cz: f32,
    dx: f32, dy: f32, dz: f32,
    ex: f32, ey: f32, ez: f32,
) -> vec2<f32> {
    let aex = ax - ex; let aey = ay - ey; let aez = az - ez;
    let bex = bx - ex; let bey = by - ey; let bez = bz - ez;
    let cex = cx - ex; let cey = cy - ey; let cez = cz - ez;
    let dex = dx - ex; let dey = dy - ey; let dez = dz - ez;

    let ab = aex * bey - bex * aey;
    let bc = bex * cey - cex * bey;
    let cd = cex * dey - dex * cey;
    let da = dex * aey - aex * dey;
    let ac = aex * cey - cex * aey;
    let bd = bex * dey - dex * bey;

    let abc = aez * bc - bez * ac + cez * ab;
    let bcd = bez * cd - cez * bd + dez * bc;
    let cda = cez * da + dez * ac + aez * cd;
    let dab = dez * ab + aez * bd + bez * da;

    let alift = aex * aex + aey * aey + aez * aez;
    let blift = bex * bex + bey * bey + bez * bez;
    let clift = cex * cex + cey * cey + cez * cez;
    let dlift = dex * dex + dey * dey + dez * dez;

    let det = (dlift * abc - clift * dab) + (blift * cda - alift * bcd);

    let permanent = abs(dlift * abc) + abs(clift * dab)
                  + abs(blift * cda) + abs(alift * bcd);
    let err_bound = 1.5e-5 * permanent;

    if abs(det) > err_bound {
        return vec2<f32>(det, 0.0);
    }
    return vec2<f32>(det, 1.0);
}

fn insphere_exact(
    ax: f32, ay: f32, az: f32,
    bx: f32, by: f32, bz: f32,
    cx: f32, cy: f32, cz: f32,
    dx: f32, dy: f32, dz: f32,
    ex: f32, ey: f32, ez: f32,
) -> i32 {
    let fast = insphere_fast(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, ex, ey, ez);
    if fast.y == 0.0 {
        if fast.x > 0.0 { return 1; }
        if fast.x < 0.0 { return -1; }
        return 0;
    }
    return insphere_dd(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, ex, ey, ez);
}

// --- Wrapper functions with SoS tie-breaking ---

fn insphere_with_sos(
    a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>, e: vec3<f32>,
    va: u32, vb: u32, vc: u32, vd: u32, ve: u32
) -> i32 {
    let result = insphere_exact(
        a.x, a.y, a.z,
        b.x, b.y, b.z,
        c.x, c.y, c.z,
        d.x, d.y, d.z,
        e.x, e.y, e.z
    );

    if result == 0 {
        return sos_insphere_index(va, vb, vc, vd, ve);
    }

    return result;
}

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
fn check_delaunay_exact(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let act_tet_num = params.x;
    let vote_offset = params.y;

    if idx >= act_tet_num {
        return;
    }

    vote_arr[idx] = -1;
    let bot_ti = act_tet_vec[idx];

    if bot_ti < 0 {
        return;
    }

    let bot_ti_u = u32(bot_ti);

    if !is_tet_alive(tet_info[bot_ti_u]) {
        // Mark as dead
        act_tet_vec[idx] = -1;
        return;
    }

    // Load bot tet opposite info and neighbor vertices
    var bot_opp = array<u32, 4>(
        atomicLoad(&tet_opp[bot_ti_u * 4u + 0u]),
        atomicLoad(&tet_opp[bot_ti_u * 4u + 1u]),
        atomicLoad(&tet_opp[bot_ti_u * 4u + 2u]),
        atomicLoad(&tet_opp[bot_ti_u * 4u + 3u]),
    );

    var opp_vert = array<i32, 4>(-1, -1, -1, -1);

    // Load opposite vertices
    for (var bot_vi = 0u; bot_vi < 4u; bot_vi++) {
        var top_vert = -1;

        if !is_opp_internal(bot_opp[bot_vi]) {
            let top_ti = decode_opp_tet(bot_opp[bot_vi]);
            let top_vi = decode_opp_vi(bot_opp[bot_vi]);
            let tet = tets[top_ti];
            // Avoid dynamic vector indexing - use select chain
            let tet_elem = select(select(tet.x, tet.y, top_vi == 1u), select(tet.z, tet.w, top_vi == 3u), top_vi >= 2u);
            top_vert = i32(tet_elem);

            // If neighbor changed and has lower index, mark as negative
            if (top_ti < bot_ti_u) && is_tet_changed(tet_info[top_ti]) {
                top_vert = make_negative(top_vert);
            }
        }

        opp_vert[bot_vi] = top_vert;
    }

    // Check flipping configuration
    var check_vi = 1u;

    for (var bot_vi = 0u; bot_vi < 4u; bot_vi++) {
        let top_vert = opp_vert[bot_vi];

        if top_vert < 0 {
            continue;
        }

        // Check for 3-2 flip
        let bot_ord_vi = TET_VI_AS_SEEN_FROM[bot_vi];
        var i = 0u;

        for (; i < 3u; i++) {
            // Avoid dynamic vector indexing
            let ord_vi_elem = select(bot_ord_vi.x, select(bot_ord_vi.y, bot_ord_vi.z, i == 2u), i >= 1u);
            let side_vert = opp_vert[ord_vi_elem];

            // More than 3 tets around edge
            if side_vert != top_vert && side_vert != make_negative(top_vert) {
                continue;
            }

            // 3-2 flip is possible
            break;
        }

        check_vi = (check_vi << 4u) | bot_vi | (i << 2u);
    }

    if check_vi == 1u {
        return; // Nothing to check
    }

    // Do sphere check
    let bot_tet = tets[bot_ti_u];
    let bot_p = array<vec3<f32>, 4>(
        points[bot_tet.x].xyz,
        points[bot_tet.y].xyz,
        points[bot_tet.z].xyz,
        points[bot_tet.w].xyz,
    );

    var check_23 = 1u;
    var has_flip = false;

    // Check 2-3 flips
    var check_vi_local = check_vi;
    loop {
        if check_vi_local <= 1u {
            break;
        }

        let bot_vi = check_vi_local & 3u;
        let bot_cor_ord_vi = (check_vi_local >> 2u) & 3u;
        let top_vert_i = opp_vert[bot_vi];
        let top_vert = u32(top_vert_i);
        let top_p = points[top_vert].xyz;

        // Use exact insphere with SoS
        let side = insphere_with_sos(
            bot_p[0], bot_p[1], bot_p[2], bot_p[3], top_p,
            bot_tet.x, bot_tet.y, bot_tet.z, bot_tet.w, top_vert
        );

        if side != 1 {
            check_vi_local = check_vi_local >> 4u;
            continue; // No insphere failure
        }

        // We have insphere failure - set OPP_SPHERE_FAIL flag (CUDA: KerPredicates.cu:584)
        let old_opp = atomicLoad(&tet_opp[bot_ti_u * 4u + bot_vi]);
        atomicOr(&tet_opp[bot_ti_u * 4u + bot_vi], OPP_SPHERE_FAIL);

        if bot_cor_ord_vi < 3u {
            // 3-2 flip confirmed
            let flip_info = make_flip(bot_vi, bot_cor_ord_vi);
            vote_arr[idx] = make_vote_val(bot_ti_u, flip_info);

            let bot_ord_vi = TET_VI_AS_SEEN_FROM[bot_vi];
            // Avoid dynamic vector indexing
            let bot_cor_vi = select(bot_ord_vi.x, select(bot_ord_vi.y, bot_ord_vi.z, bot_cor_ord_vi == 2u), bot_cor_ord_vi >= 1u);
            let bot_opp_ti = decode_opp_tet(bot_opp[bot_cor_vi]);
            let top_ti = decode_opp_tet(bot_opp[bot_vi]);
            vote_for_flip32(vote_offset, bot_ti_u, top_ti, bot_opp_ti);

            has_flip = true;
            check_23 = 1u; // No more need to check 2-3
            break;
        }

        // Postpone check for 2-3 flippability
        check_23 = (check_23 << 2u) | bot_vi;
        check_vi_local = check_vi_local >> 4u;
    }

    // Try for 2-3 flip
    loop {
        if check_23 <= 1u {
            break;
        }

        let bot_vi = check_23 & 3u;
        let top_vert_i = opp_vert[bot_vi];
        let top_vert = u32(top_vert_i);
        let top_p = points[top_vert].xyz;
        let bot_ord_vi = TET_VI_AS_SEEN_FROM[bot_vi];

        has_flip = true;

        // Go around bottom-top tetra, check 3 sides
        for (var i = 0u; i < 3u; i++) {
            // Avoid dynamic vector and array indexing
            let ord_vi_elem = select(bot_ord_vi.x, select(bot_ord_vi.y, bot_ord_vi.z, i == 2u), i >= 1u);
            // CRITICAL FIX: Cannot use TET_VI_AS_SEEN_FROM[variable] - causes SIGSEGV
            let fv = get_tet_vi_as_seen_from(ord_vi_elem);

            // Extract bot_p elements using select to avoid dynamic array indexing
            let p0 = select(select(bot_p[0], bot_p[1], fv.x == 1u), select(bot_p[2], bot_p[3], fv.x == 3u), fv.x >= 2u);
            let p1 = select(select(bot_p[0], bot_p[1], fv.y == 1u), select(bot_p[2], bot_p[3], fv.y == 3u), fv.y >= 2u);
            let p2 = select(select(bot_p[0], bot_p[1], fv.z == 1u), select(bot_p[2], bot_p[3], fv.z == 3u), fv.z >= 2u);

            // Extract vertex indices for SoS
            let va = select(select(bot_tet.x, bot_tet.y, fv.x == 1u), select(bot_tet.z, bot_tet.w, fv.x == 3u), fv.x >= 2u);
            let vb = select(select(bot_tet.x, bot_tet.y, fv.y == 1u), select(bot_tet.z, bot_tet.w, fv.y == 3u), fv.y >= 2u);
            let vc = select(select(bot_tet.x, bot_tet.y, fv.z == 1u), select(bot_tet.z, bot_tet.w, fv.z == 3u), fv.z >= 2u);

            // Use exact orient3d with SoS
            let ort = orient3d_with_sos(p0, p1, p2, top_p, va, vb, vc, top_vert);

            if ort != 1 {
                has_flip = false;
                break; // Cannot do 2-3 flip
            }
        }

        if has_flip {
            // 2-3 flip possible!
            let flip_info = make_flip(bot_vi, 3u);
            vote_arr[idx] = make_vote_val(bot_ti_u, flip_info);
            let top_ti = decode_opp_tet(bot_opp[bot_vi]);
            vote_for_flip23(vote_offset, bot_ti_u, top_ti);
            break;
        }

        check_23 = check_23 >> 2u;
    }

    // Note: No need to store updated opp info - exact predicates don't set flags

    // Reset flip counter (done by thread 0 of block 0)
    if gid.x == 0u {
        atomicStore(&counters[COUNTER_FLIP], 0u);
    }
}
