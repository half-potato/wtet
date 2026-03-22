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
@group(0) @binding(8) var<uniform> params: vec4<u32>; // x = act_tet_num, y = vote_offset, z = inf_idx

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

// Direct array indexing (now supported in wgpu 28+)
fn get_tet_vi_as_seen_from(vi: u32) -> vec3<u32> {
    return TET_VI_AS_SEEN_FROM[vi];
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
    // Bit 2 = internal flag (CUDA: CommonTypes.h:322-324)
    // Internal faces are between split sibling tets — already locally Delaunay.
    return (opp & 4u) != 0u;
}

// CUDA: makeNegative(v) = -(v + 2), escapes -1 sentinel (KerCommon.h:192-195)
fn make_negative(val: i32) -> i32 {
    return -(val + 2);
}

fn make_positive(val: i32) -> i32 {
    return -(val + 2);
}

fn tet_vertex_val(tet: vec4<u32>, i: u32) -> u32 {
    if i == 0u { return tet.x; }
    else if i == 1u { return tet.y; }
    else if i == 2u { return tet.z; }
    else { return tet.w; }
}

fn orient3d_simple(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> i32 {
    let ad = a - d;
    let bd = b - d;
    let cd = c - d;
    let det = ad.x * (bd.y * cd.z - bd.z * cd.y)
            + bd.x * (cd.y * ad.z - cd.z * ad.y)
            + cd.x * (ad.y * bd.z - ad.z * bd.y);
    if det > 0.0 { return 1; }
    else if det < 0.0 { return -1; }
    return 0;
}

fn make_flip(bot_vi: u32, bot_cor_ord_vi: u32) -> u32 {
    return (bot_cor_ord_vi << 2u) | bot_vi;
}

fn make_vote_val(bot_ti: u32, flip_info: u32) -> i32 {
    return i32((bot_ti << 8u) | flip_info);
}

fn vote_for_flip23(vote_offset: u32, bot_ti: u32, top_ti: u32) {
    // CUDA: voteVal = voteOffset + botTi; atomicMin(&tetVoteArr[botTi], voteVal)
    let vote_val = i32(vote_offset + bot_ti);
    atomicMin(&tet_vote_arr[bot_ti], vote_val);
    atomicMin(&tet_vote_arr[top_ti], vote_val);
}

fn vote_for_flip32(vote_offset: u32, bot_ti: u32, top_ti: u32, bot_opp_ti: u32) {
    // CUDA: voteVal = voteOffset + botTi; atomicMin(&tetVoteArr[botTi/topTi/sideTi], voteVal)
    let vote_val = i32(vote_offset + bot_ti);
    atomicMin(&tet_vote_arr[bot_ti], vote_val);
    atomicMin(&tet_vote_arr[top_ti], vote_val);
    atomicMin(&tet_vote_arr[bot_opp_ti], vote_val);
}

// --- SoS (Simulation of Simplicity) tie-breaking ---

// Exact orient2d using DD: det = (pa[0]-pc[0])*(pb[1]-pc[1]) - (pa[1]-pc[1])*(pb[0]-pc[0])
fn orient2d_exact_2(pa0: f32, pa1: f32, pb0: f32, pb1: f32, pc0: f32, pc1: f32) -> f32 {
    let t1 = dd_mul(dd_from_f32(pa0 - pc0), dd_from_f32(pb1 - pc1));
    let t2 = dd_mul(dd_from_f32(pa1 - pc1), dd_from_f32(pb0 - pc0));
    let det = dd_sub(t1, t2);
    if det.hi != 0.0 { return det.hi; }
    return det.lo;
}

// --- Lifted predicate helpers for insphere SoS ---
// CUDA: orient3dFastExact_Lifted (KerShewchuk.h:2016)
// When lifted=true, z-column replaced by x²+y²+z² (squared distance)
// For non-lifted: standard orient3d on (col0, col1, col2)
// For lifted: orient3d on (col0, col1, lift) where lift = col0²-col1²+col2²
// Uses DD arithmetic (approximate for lifted, but sufficient for f32 grid inputs)

// orient3d_lifted_dd: 4-point orient3d, optionally with lifted z-coordinate
// When lifted=false: det of (a0-d0, a1-d1, a2-d2) rows for each point
// When lifted=true: z replaced by x²+y²+z² for each point
fn orient3d_lifted_dd(
    a0: f32, a1: f32, a2: f32,
    b0: f32, b1: f32, b2: f32,
    c0: f32, c1: f32, c2: f32,
    d0: f32, d1: f32, d2: f32,
    lifted: bool
) -> i32 {
    let adx = dd_from_f32(a0 - d0);
    let ady = dd_from_f32(a1 - d1);
    let bdx = dd_from_f32(b0 - d0);
    let bdy = dd_from_f32(b1 - d1);
    let cdx = dd_from_f32(c0 - d0);
    let cdy = dd_from_f32(c1 - d1);

    var adz: DD;
    var bdz: DD;
    var cdz: DD;
    if lifted {
        // z = x²+y²+z² (lifted coordinate = squared Euclidean norm)
        // CUDA: adz = adx*adx + ady*ady + adz*adz (KerShewchuk.h:2044)
        let adz_raw = dd_from_f32(a2 - d2);
        adz = dd_add(dd_add(dd_mul(adx, adx), dd_mul(ady, ady)), dd_mul(adz_raw, adz_raw));
        let bdz_raw = dd_from_f32(b2 - d2);
        bdz = dd_add(dd_add(dd_mul(bdx, bdx), dd_mul(bdy, bdy)), dd_mul(bdz_raw, bdz_raw));
        let cdz_raw = dd_from_f32(c2 - d2);
        cdz = dd_add(dd_add(dd_mul(cdx, cdx), dd_mul(cdy, cdy)), dd_mul(cdz_raw, cdz_raw));
    } else {
        adz = dd_from_f32(a2 - d2);
        bdz = dd_from_f32(b2 - d2);
        cdz = dd_from_f32(c2 - d2);
    }

    let t1 = dd_sub(dd_mul(bdy, cdz), dd_mul(bdz, cdy));
    let t2 = dd_sub(dd_mul(cdy, adz), dd_mul(cdz, ady));
    let t3 = dd_sub(dd_mul(ady, bdz), dd_mul(adz, bdy));
    let det = dd_add(dd_add(dd_mul(adx, t1), dd_mul(bdx, t2)), dd_mul(cdx, t3));
    return dd_sign(det);
}

// orient2d_lifted_dd: 3-point orient2d, optionally with lifted 2nd coordinate
// When lifted=false: det of 2x2 matrix [(a0-c0, a1-c1), (b0-b0, b1-c1)]
// When lifted=true: col1 replaced by x²-y²+z² where x=col0, y=col1, z=col2
// CUDA: orient2dExact_Lifted (KerShewchuk.h:1801)
fn orient2d_lifted_dd(
    a0: f32, a1: f32, a2: f32,
    b0: f32, b1: f32, b2: f32,
    c0: f32, c1: f32, c2: f32,
    lifted: bool
) -> i32 {
    if !lifted {
        // Standard orient2d: (a0-c0)*(b1-c1) - (a1-c1)*(b0-c0)
        let r = orient2d_exact_2(a0, a1, b0, b1, c0, c1);
        if r > 0.0 { return 1; }
        if r < 0.0 { return -1; }
        return 0;
    }
    // Lifted: replace col1 with lift = col0² - col1² + col2²
    // CUDA computes: palift = pa[0]² - pa[1]² + pa[2]² (exact expansion)
    // We use DD: lift_a = a0*a0 - a1*a1 + a2*a2
    let la = dd_add(dd_sub(dd_mul(dd_from_f32(a0), dd_from_f32(a0)),
                           dd_mul(dd_from_f32(a1), dd_from_f32(a1))),
                    dd_mul(dd_from_f32(a2), dd_from_f32(a2)));
    let lb = dd_add(dd_sub(dd_mul(dd_from_f32(b0), dd_from_f32(b0)),
                           dd_mul(dd_from_f32(b1), dd_from_f32(b1))),
                    dd_mul(dd_from_f32(b2), dd_from_f32(b2)));
    let lc = dd_add(dd_sub(dd_mul(dd_from_f32(c0), dd_from_f32(c0)),
                           dd_mul(dd_from_f32(c1), dd_from_f32(c1))),
                    dd_mul(dd_from_f32(c2), dd_from_f32(c2)));
    // orient2d with (col0, lift): det = (a0-c0)*(lb-lc) - (la-lc)*(b0-c0)
    let t1 = dd_mul(dd_from_f32(a0 - c0), dd_sub(lb, lc));
    let t2 = dd_mul(dd_sub(la, lc), dd_from_f32(b0 - c0));
    let det = dd_sub(t1, t2);
    return dd_sign(det);
}

// orient1d_lifted_dd: 2-point comparison, optionally lifted
// When lifted=false: sign of (a0 - b0)
// When lifted=true: sign of (a.x²+a.y²+a.z² - b.x²-b.y²-b.z²)
// CUDA: orient1dExact_Lifted (KerShewchuk.h:1751)
fn orient1d_lifted_dd(
    a0: f32, a1: f32, a2: f32,
    b0: f32, b1: f32, b2: f32,
    lifted: bool
) -> i32 {
    if !lifted {
        if a0 > b0 { return 1; }
        if a0 < b0 { return -1; }
        return 0;
    }
    // Lifted: (a0²+a1²+a2²) - (b0²+b1²+b2²)
    let aa = dd_add(dd_add(dd_mul(dd_from_f32(a0), dd_from_f32(a0)),
                           dd_mul(dd_from_f32(a1), dd_from_f32(a1))),
                    dd_mul(dd_from_f32(a2), dd_from_f32(a2)));
    let bb = dd_add(dd_add(dd_mul(dd_from_f32(b0), dd_from_f32(b0)),
                           dd_mul(dd_from_f32(b1), dd_from_f32(b1))),
                    dd_mul(dd_from_f32(b2), dd_from_f32(b2)));
    return dd_sign(dd_sub(aa, bb));
}

// Port of CUDA doOrient3DSoSOnly (KerPredWrapper.h:131-235)
// Edelsbrunner-Mücke SoS perturbation scheme for orient3d
// Called when exact orient3d returns 0 (degenerate configuration)
fn sos_orient3d(
    va: u32, vb: u32, vc: u32, vd: u32,
    pa: vec3<f32>, pb: vec3<f32>, pc: vec3<f32>, pd: vec3<f32>
) -> i32 {
    // Sort vertices by index, track parity
    var v = array<u32, 4>(va, vb, vc, vd);
    var p = array<vec3<f32>, 4>(pa, pb, pc, pd);
    var pn = 1;

    // Sorting network for 4 elements (matches CUDA's swap sequence)
    if v[0] > v[2] { let tv = v[0]; v[0] = v[2]; v[2] = tv; let tp = p[0]; p[0] = p[2]; p[2] = tp; pn = -pn; }
    if v[1] > v[3] { let tv = v[1]; v[1] = v[3]; v[3] = tv; let tp = p[1]; p[1] = p[3]; p[3] = tp; pn = -pn; }
    if v[0] > v[1] { let tv = v[0]; v[0] = v[1]; v[1] = tv; let tp = p[0]; p[0] = p[1]; p[1] = tp; pn = -pn; }
    if v[2] > v[3] { let tv = v[2]; v[2] = v[3]; v[3] = tv; let tp = p[2]; p[2] = p[3]; p[3] = tp; pn = -pn; }
    if v[1] > v[2] { let tv = v[1]; v[1] = v[2]; v[2] = tv; let tp = p[1]; p[1] = p[2]; p[2] = tp; pn = -pn; }

    // Cascade through sub-determinants (14 levels)
    // Each level computes an increasingly lower-dimensional predicate
    var result: f32 = 0.0;

    // Depth 0: orient2d(p[1],p[2],p[3]) on (x,y)
    result = orient2d_exact_2(p[1].x, p[1].y, p[2].x, p[2].y, p[3].x, p[3].y);
    if result != 0.0 { return select(-1, 1, (result * f32(pn)) > 0.0); }

    // Depth 1: orient2d(p[1],p[2],p[3]) on (x,z) — negate
    result = orient2d_exact_2(p[1].x, p[1].z, p[2].x, p[2].z, p[3].x, p[3].z);
    if result != 0.0 { return select(-1, 1, (-result * f32(pn)) > 0.0); }

    // Depth 2: orient2d(p[1],p[2],p[3]) on (y,z)
    result = orient2d_exact_2(p[1].y, p[1].z, p[2].y, p[2].z, p[3].y, p[3].z);
    if result != 0.0 { return select(-1, 1, (result * f32(pn)) > 0.0); }

    // Depth 3: orient2d(p[0],p[2],p[3]) on (x,y) — negate
    result = orient2d_exact_2(p[0].x, p[0].y, p[2].x, p[2].y, p[3].x, p[3].y);
    if result != 0.0 { return select(-1, 1, (-result * f32(pn)) > 0.0); }

    // Depth 4: p[2].x - p[3].x
    result = p[2].x - p[3].x;
    if result != 0.0 { return select(-1, 1, (result * f32(pn)) > 0.0); }

    // Depth 5: p[2].y - p[3].y — negate
    result = p[2].y - p[3].y;
    if result != 0.0 { return select(-1, 1, (-result * f32(pn)) > 0.0); }

    // Depth 6: orient2d(p[0],p[2],p[3]) on (x,z)
    result = orient2d_exact_2(p[0].x, p[0].z, p[2].x, p[2].z, p[3].x, p[3].z);
    if result != 0.0 { return select(-1, 1, (result * f32(pn)) > 0.0); }

    // Depth 7: p[2].z - p[3].z
    result = p[2].z - p[3].z;
    if result != 0.0 { return select(-1, 1, (result * f32(pn)) > 0.0); }

    // Depth 8: orient2d(p[0],p[2],p[3]) on (y,z) — negate
    result = orient2d_exact_2(p[0].y, p[0].z, p[2].y, p[2].z, p[3].y, p[3].z);
    if result != 0.0 { return select(-1, 1, (-result * f32(pn)) > 0.0); }

    // Depth 9: orient2d(p[0],p[1],p[3]) on (x,y)
    result = orient2d_exact_2(p[0].x, p[0].y, p[1].x, p[1].y, p[3].x, p[3].y);
    if result != 0.0 { return select(-1, 1, (result * f32(pn)) > 0.0); }

    // Depth 10: p[1].x - p[3].x — negate
    result = p[1].x - p[3].x;
    if result != 0.0 { return select(-1, 1, (-result * f32(pn)) > 0.0); }

    // Depth 11: p[1].y - p[3].y
    result = p[1].y - p[3].y;
    if result != 0.0 { return select(-1, 1, (result * f32(pn)) > 0.0); }

    // Depth 12: p[0].x - p[3].x
    result = p[0].x - p[3].x;
    if result != 0.0 { return select(-1, 1, (result * f32(pn)) > 0.0); }

    // Depth 13: always 1.0 (guaranteed non-zero)
    return select(-1, 1, f32(pn) > 0.0);
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

// --- SoS insphere: Edelsbrunner-Mücke 49-level cascade ---
// Port of CUDA doInSphereSoSOnly (KerPredWrapper.h:314-777)
// Called when exact insphere returns 0 (degenerate configuration)
fn sos_insphere(
    va: u32, vb: u32, vc: u32, vd: u32, ve: u32,
    pa: vec3<f32>, pb: vec3<f32>, pc: vec3<f32>, pd: vec3<f32>, pe: vec3<f32>
) -> i32 {
    // Sort 5 vertices by index, track coordinates with them
    // 9-swap sorting network (CUDA: KerPredWrapper.h:326-334)
    var v = array<u32, 5>(va, vb, vc, vd, ve);
    var p = array<vec3<f32>, 5>(pa, pb, pc, pd, pe);
    var sc = 0;

    if v[0] > v[4] { let tv=v[0]; v[0]=v[4]; v[4]=tv; let tp=p[0]; p[0]=p[4]; p[4]=tp; sc++; }
    if v[1] > v[3] { let tv=v[1]; v[1]=v[3]; v[3]=tv; let tp=p[1]; p[1]=p[3]; p[3]=tp; sc++; }
    if v[0] > v[2] { let tv=v[0]; v[0]=v[2]; v[2]=tv; let tp=p[0]; p[0]=p[2]; p[2]=tp; sc++; }
    if v[2] > v[4] { let tv=v[2]; v[2]=v[4]; v[4]=tv; let tp=p[2]; p[2]=p[4]; p[4]=tp; sc++; }
    if v[0] > v[1] { let tv=v[0]; v[0]=v[1]; v[1]=tv; let tp=p[0]; p[0]=p[1]; p[1]=tp; sc++; }
    if v[2] > v[3] { let tv=v[2]; v[2]=v[3]; v[3]=tv; let tp=p[2]; p[2]=p[3]; p[3]=tp; sc++; }
    if v[1] > v[4] { let tv=v[1]; v[1]=v[4]; v[4]=tv; let tp=p[1]; p[1]=p[4]; p[4]=tp; sc++; }
    if v[1] > v[2] { let tv=v[1]; v[1]=v[2]; v[2]=tv; let tp=p[1]; p[1]=p[2]; p[2]=tp; sc++; }
    if v[3] > v[4] { let tv=v[3]; v[3]=v[4]; v[4]=tv; let tp=p[3]; p[3]=p[4]; p[4]=tp; sc++; }

    // ip[i] = p[i], indexed as ip[i].x/y/z = ip[i][0]/[1]/[2]

    var orient: i32;

    // Depth 1: orient3d(ip[1],ip[2],ip[3],ip[4]) — non-lifted
    orient = orient3d_lifted_dd(p[1].x,p[1].y,p[1].z, p[2].x,p[2].y,p[2].z, p[3].x,p[3].y,p[3].z, p[4].x,p[4].y,p[4].z, false);
    if orient != 0 { return sos_apply(orient, sc, true); }

    // Depth 2: orient3d(ip[1],ip[2],ip[3],ip[4]) — lifted, cols(x,y,z)
    orient = orient3d_lifted_dd(p[1].x,p[1].y,p[1].z, p[2].x,p[2].y,p[2].z, p[3].x,p[3].y,p[3].z, p[4].x,p[4].y,p[4].z, true);
    if orient != 0 { return sos_apply(orient, sc, false); }

    // Depth 3: orient3d lifted, cols(x,z,y) — note swapped y/z in cols 1,2
    orient = orient3d_lifted_dd(p[1].x,p[1].z,p[1].y, p[2].x,p[2].z,p[2].y, p[3].x,p[3].z,p[3].y, p[4].x,p[4].z,p[4].y, true);
    if orient != 0 { return sos_apply(orient, sc, true); }

    // Depth 4: orient3d lifted, cols(y,z,x) — swapped
    orient = orient3d_lifted_dd(p[1].y,p[1].z,p[1].x, p[2].y,p[2].z,p[2].x, p[3].y,p[3].z,p[3].x, p[4].y,p[4].z,p[4].x, true);
    if orient != 0 { return sos_apply(orient, sc, false); }

    // Depth 5: orient3d(ip[0],ip[2],ip[3],ip[4]) — non-lifted
    orient = orient3d_lifted_dd(p[0].x,p[0].y,p[0].z, p[2].x,p[2].y,p[2].z, p[3].x,p[3].y,p[3].z, p[4].x,p[4].y,p[4].z, false);
    if orient != 0 { return sos_apply(orient, sc, false); }

    // Depth 6: orient2d(ip[2],ip[3],ip[4]) cols(x,y) — non-lifted
    orient = orient2d_lifted_dd(p[2].x,p[2].y,p[2].z, p[3].x,p[3].y,p[3].z, p[4].x,p[4].y,p[4].z, false);
    if orient != 0 { return sos_apply(orient, sc, false); }

    // Depth 7: orient2d cols(x,z) — non-lifted (note: reuses ip[2..4] with col1=z)
    orient = orient2d_lifted_dd(p[2].x,p[2].z,p[2].y, p[3].x,p[3].z,p[3].y, p[4].x,p[4].z,p[4].y, false);
    if orient != 0 { return sos_apply(orient, sc, true); }

    // Depth 8: orient2d cols(y,z) — non-lifted
    orient = orient2d_lifted_dd(p[2].y,p[2].z,p[2].x, p[3].y,p[3].z,p[3].x, p[4].y,p[4].z,p[4].x, false);
    if orient != 0 { return sos_apply(orient, sc, false); }

    // Depth 9: orient3d(ip[0],ip[2],ip[3],ip[4]) — lifted
    orient = orient3d_lifted_dd(p[0].x,p[0].y,p[0].z, p[2].x,p[2].y,p[2].z, p[3].x,p[3].y,p[3].z, p[4].x,p[4].y,p[4].z, true);
    if orient != 0 { return sos_apply(orient, sc, true); }

    // Depth 10: orient2d(ip[2],ip[3],ip[4]) — lifted
    orient = orient2d_lifted_dd(p[2].x,p[2].y,p[2].z, p[3].x,p[3].y,p[3].z, p[4].x,p[4].y,p[4].z, true);
    if orient != 0 { return sos_apply(orient, sc, false); }

    // Depth 11: orient2d lifted, cols(y,x,z) for ip[2..4]
    orient = orient2d_lifted_dd(p[2].y,p[2].x,p[2].z, p[3].y,p[3].x,p[3].z, p[4].y,p[4].x,p[4].z, true);
    if orient != 0 { return sos_apply(orient, sc, true); }

    // Depth 12: orient3d lifted, cols(x,z,y) for ip[0],ip[2..4]
    orient = orient3d_lifted_dd(p[0].x,p[0].z,p[0].y, p[2].x,p[2].z,p[2].y, p[3].x,p[3].z,p[3].y, p[4].x,p[4].z,p[4].y, true);
    if orient != 0 { return sos_apply(orient, sc, false); }

    // Depth 13: orient2d lifted, cols(z,x,y) for ip[2..4]
    orient = orient2d_lifted_dd(p[2].z,p[2].x,p[2].y, p[3].z,p[3].x,p[3].y, p[4].z,p[4].x,p[4].y, true);
    if orient != 0 { return sos_apply(orient, sc, false); }

    // Depth 14: orient3d lifted, cols(y,z,x) for ip[0],ip[2..4]
    orient = orient3d_lifted_dd(p[0].y,p[0].z,p[0].x, p[2].y,p[2].z,p[2].x, p[3].y,p[3].z,p[3].x, p[4].y,p[4].z,p[4].x, true);
    if orient != 0 { return sos_apply(orient, sc, true); }

    // Depth 15: orient3d(ip[0],ip[1],ip[3],ip[4]) — non-lifted
    orient = orient3d_lifted_dd(p[0].x,p[0].y,p[0].z, p[1].x,p[1].y,p[1].z, p[3].x,p[3].y,p[3].z, p[4].x,p[4].y,p[4].z, false);
    if orient != 0 { return sos_apply(orient, sc, true); }

    // Depth 16: orient2d(ip[1],ip[3],ip[4]) cols(x,y) — non-lifted
    orient = orient2d_lifted_dd(p[1].x,p[1].y,p[1].z, p[3].x,p[3].y,p[3].z, p[4].x,p[4].y,p[4].z, false);
    if orient != 0 { return sos_apply(orient, sc, true); }

    // Depth 17: orient2d(ip[1],ip[3],ip[4]) cols(x,z) — non-lifted
    orient = orient2d_lifted_dd(p[1].x,p[1].z,p[1].y, p[3].x,p[3].z,p[3].y, p[4].x,p[4].z,p[4].y, false);
    if orient != 0 { return sos_apply(orient, sc, false); }

    // Depth 18: orient2d(ip[1],ip[3],ip[4]) cols(y,z) — non-lifted
    orient = orient2d_lifted_dd(p[1].y,p[1].z,p[1].x, p[3].y,p[3].z,p[3].x, p[4].y,p[4].z,p[4].x, false);
    if orient != 0 { return sos_apply(orient, sc, true); }

    // Depth 19: orient2d(ip[0],ip[3],ip[4]) cols(x,y) — non-lifted
    orient = orient2d_lifted_dd(p[0].x,p[0].y,p[0].z, p[3].x,p[3].y,p[3].z, p[4].x,p[4].y,p[4].z, false);
    if orient != 0 { return sos_apply(orient, sc, false); }

    // Depth 20: orient1d ip[3] vs ip[4], col x — non-lifted
    orient = orient1d_lifted_dd(p[3].x,p[3].y,p[3].z, p[4].x,p[4].y,p[4].z, false);
    if orient != 0 { return sos_apply(orient, sc, true); }

    // Depth 21: orient1d ip[3] vs ip[4], col y — non-lifted
    let d21 = p[3].y - p[4].y;
    if d21 != 0.0 { return sos_apply(select(-1, 1, d21 > 0.0), sc, false); }

    // Depth 22: orient2d(ip[0],ip[3],ip[4]) cols(x,z) — non-lifted
    orient = orient2d_lifted_dd(p[0].x,p[0].z,p[0].y, p[3].x,p[3].z,p[3].y, p[4].x,p[4].z,p[4].y, false);
    if orient != 0 { return sos_apply(orient, sc, true); }

    // Depth 23: orient1d ip[3] vs ip[4], col z — non-lifted
    let d23 = p[3].z - p[4].z;
    if d23 != 0.0 { return sos_apply(select(-1, 1, d23 > 0.0), sc, true); }

    // Depth 24: orient2d(ip[0],ip[3],ip[4]) cols(y,z) — non-lifted
    orient = orient2d_lifted_dd(p[0].y,p[0].z,p[0].x, p[3].y,p[3].z,p[3].x, p[4].y,p[4].z,p[4].x, false);
    if orient != 0 { return sos_apply(orient, sc, false); }

    // Depth 25: orient3d(ip[0],ip[1],ip[3],ip[4]) — lifted
    orient = orient3d_lifted_dd(p[0].x,p[0].y,p[0].z, p[1].x,p[1].y,p[1].z, p[3].x,p[3].y,p[3].z, p[4].x,p[4].y,p[4].z, true);
    if orient != 0 { return sos_apply(orient, sc, false); }

    // Depth 26: orient2d(ip[1],ip[3],ip[4]) — lifted
    orient = orient2d_lifted_dd(p[1].x,p[1].y,p[1].z, p[3].x,p[3].y,p[3].z, p[4].x,p[4].y,p[4].z, true);
    if orient != 0 { return sos_apply(orient, sc, true); }

    // Depth 27: orient2d lifted, cols(y,x,z) for ip[1],ip[3],ip[4]
    orient = orient2d_lifted_dd(p[1].y,p[1].x,p[1].z, p[3].y,p[3].x,p[3].z, p[4].y,p[4].x,p[4].z, true);
    if orient != 0 { return sos_apply(orient, sc, false); }

    // Depth 28: orient2d(ip[0],ip[3],ip[4]) — lifted
    orient = orient2d_lifted_dd(p[0].x,p[0].y,p[0].z, p[3].x,p[3].y,p[3].z, p[4].x,p[4].y,p[4].z, true);
    if orient != 0 { return sos_apply(orient, sc, false); }

    // Depth 29: orient1d ip[3] vs ip[4] — lifted
    orient = orient1d_lifted_dd(p[3].x,p[3].y,p[3].z, p[4].x,p[4].y,p[4].z, true);
    if orient != 0 { return sos_apply(orient, sc, false); }

    // Depth 30: orient2d lifted, cols(y,x,z) for ip[0],ip[3],ip[4]
    orient = orient2d_lifted_dd(p[0].y,p[0].x,p[0].z, p[3].y,p[3].x,p[3].z, p[4].y,p[4].x,p[4].z, true);
    if orient != 0 { return sos_apply(orient, sc, true); }

    // Depth 31: orient3d lifted, cols(x,z,y) for ip[0],ip[1],ip[3],ip[4]
    orient = orient3d_lifted_dd(p[0].x,p[0].z,p[0].y, p[1].x,p[1].z,p[1].y, p[3].x,p[3].z,p[3].y, p[4].x,p[4].z,p[4].y, true);
    if orient != 0 { return sos_apply(orient, sc, true); }

    // Depth 32: orient2d lifted, cols(z,x,y) for ip[1],ip[3],ip[4]
    orient = orient2d_lifted_dd(p[1].z,p[1].x,p[1].y, p[3].z,p[3].x,p[3].y, p[4].z,p[4].x,p[4].y, true);
    if orient != 0 { return sos_apply(orient, sc, true); }

    // Depth 33: orient2d lifted, cols(z,x,y) for ip[0],ip[3],ip[4]
    orient = orient2d_lifted_dd(p[0].z,p[0].x,p[0].y, p[3].z,p[3].x,p[3].y, p[4].z,p[4].x,p[4].y, true);
    if orient != 0 { return sos_apply(orient, sc, false); }

    // Depth 34: orient3d lifted, cols(y,z,x) for ip[0],ip[1],ip[3],ip[4]
    orient = orient3d_lifted_dd(p[0].y,p[0].z,p[0].x, p[1].y,p[1].z,p[1].x, p[3].y,p[3].z,p[3].x, p[4].y,p[4].z,p[4].x, true);
    if orient != 0 { return sos_apply(orient, sc, false); }

    // Depth 35: orient3d(ip[0],ip[1],ip[2],ip[4]) — non-lifted
    orient = orient3d_lifted_dd(p[0].x,p[0].y,p[0].z, p[1].x,p[1].y,p[1].z, p[2].x,p[2].y,p[2].z, p[4].x,p[4].y,p[4].z, false);
    if orient != 0 { return sos_apply(orient, sc, false); }

    // Depth 36: orient2d(ip[1],ip[2],ip[4]) cols(x,y) — non-lifted
    orient = orient2d_lifted_dd(p[1].x,p[1].y,p[1].z, p[2].x,p[2].y,p[2].z, p[4].x,p[4].y,p[4].z, false);
    if orient != 0 { return sos_apply(orient, sc, false); }

    // Depth 37: orient2d(ip[1],ip[2],ip[4]) cols(x,z) — non-lifted
    orient = orient2d_lifted_dd(p[1].x,p[1].z,p[1].y, p[2].x,p[2].z,p[2].y, p[4].x,p[4].z,p[4].y, false);
    if orient != 0 { return sos_apply(orient, sc, true); }

    // Depth 38: orient2d(ip[1],ip[2],ip[4]) cols(y,z) — non-lifted
    orient = orient2d_lifted_dd(p[1].y,p[1].z,p[1].x, p[2].y,p[2].z,p[2].x, p[4].y,p[4].z,p[4].x, false);
    if orient != 0 { return sos_apply(orient, sc, false); }

    // Depth 39: orient2d(ip[0],ip[2],ip[4]) cols(x,y) — non-lifted
    orient = orient2d_lifted_dd(p[0].x,p[0].y,p[0].z, p[2].x,p[2].y,p[2].z, p[4].x,p[4].y,p[4].z, false);
    if orient != 0 { return sos_apply(orient, sc, true); }

    // Depth 40: orient1d ip[2] vs ip[4], col x — non-lifted
    let d40 = p[2].x - p[4].x;
    if d40 != 0.0 { return sos_apply(select(-1, 1, d40 > 0.0), sc, false); }

    // Depth 41: orient1d ip[2] vs ip[4], col y — non-lifted
    let d41 = p[2].y - p[4].y;
    if d41 != 0.0 { return sos_apply(select(-1, 1, d41 > 0.0), sc, true); }

    // Depth 42: orient2d(ip[0],ip[2],ip[4]) cols(x,z) — non-lifted
    orient = orient2d_lifted_dd(p[0].x,p[0].z,p[0].y, p[2].x,p[2].z,p[2].y, p[4].x,p[4].z,p[4].y, false);
    if orient != 0 { return sos_apply(orient, sc, false); }

    // Depth 43: orient1d ip[2] vs ip[4], col z — non-lifted
    let d43 = p[2].z - p[4].z;
    if d43 != 0.0 { return sos_apply(select(-1, 1, d43 > 0.0), sc, false); }

    // Depth 44: orient2d(ip[0],ip[2],ip[4]) cols(y,z) — non-lifted
    orient = orient2d_lifted_dd(p[0].y,p[0].z,p[0].x, p[2].y,p[2].z,p[2].x, p[4].y,p[4].z,p[4].x, false);
    if orient != 0 { return sos_apply(orient, sc, true); }

    // Depth 45: orient2d(ip[0],ip[1],ip[4]) cols(x,y) — non-lifted
    orient = orient2d_lifted_dd(p[0].x,p[0].y,p[0].z, p[1].x,p[1].y,p[1].z, p[4].x,p[4].y,p[4].z, false);
    if orient != 0 { return sos_apply(orient, sc, false); }

    // Depth 46: orient1d ip[1] vs ip[4], col x — non-lifted
    let d46 = p[1].x - p[4].x;
    if d46 != 0.0 { return sos_apply(select(-1, 1, d46 > 0.0), sc, true); }

    // Depth 47: orient1d ip[1] vs ip[4], col y — non-lifted
    let d47 = p[1].y - p[4].y;
    if d47 != 0.0 { return sos_apply(select(-1, 1, d47 > 0.0), sc, false); }

    // Depth 48: orient1d ip[0] vs ip[4], col x — non-lifted
    let d48 = p[0].x - p[4].x;
    if d48 != 0.0 { return sos_apply(select(-1, 1, d48 > 0.0), sc, false); }

    // Depth 49: always +1 (guaranteed non-zero terminal case)
    return sos_apply(1, sc, false);
}

// Apply depth-based sign flip and swap parity
fn sos_apply(orient: i32, swap_count: i32, negate: bool) -> i32 {
    var r = orient;
    if negate { r = -r; }
    if (swap_count & 1) != 0 { r = -r; }
    return r;
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
        return sos_insphere(va, vb, vc, vd, ve, a, b, c, d, e);
    }

    return result;
}

fn orient3d_with_sos(
    a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>,
    va: u32, vb: u32, vc: u32, vd: u32
) -> i32 {
    let result = orient3d_exact(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z, d.x, d.y, d.z);

    if result == 0 {
        // Use proper Edelsbrunner-Mücke SoS perturbation (port of CUDA doOrient3DSoSOnly)
        return sos_orient3d(va, vb, vc, vd, a, b, c, d);
    }

    // CUDA: ortToOrient negates: det < 0 → OrientPos (+1), det > 0 → OrientNeg (-1)
    // Our orient3d_exact returns Shewchuk sign directly (det < 0 → -1, det > 0 → +1)
    return result;
}

@compute @workgroup_size(256)
fn check_delaunay_exact(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let act_tet_num = params.x;
    let vote_offset = params.y;
    let inf_idx = params.z;

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

        // CRITICAL GUARD: Skip boundary faces (CUDA: KerPredicates.cu:471)
        if top_vert < 0 {
            continue;
        }

        // CRITICAL GUARD: Skip infinity vertex - prevents out-of-bounds access
        // CUDA: KerPredWrapper.h:781-801 - if (vert == _infIdx) return SideOut;
        if u32(top_vert) >= inf_idx {
            continue;  // Treat as no violation (SideOut)
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

    // Check if bot_tet is a boundary tet (contains infinity vertex)
    // CUDA: doInSphereSoS handles boundary tets with orient3d fallback (KerPredWrapper.h:818-828)
    var inf_vi: i32 = -1;
    if bot_tet.x >= inf_idx { inf_vi = 0; }
    else if bot_tet.y >= inf_idx { inf_vi = 1; }
    else if bot_tet.z >= inf_idx { inf_vi = 2; }
    else if bot_tet.w >= inf_idx { inf_vi = 3; }
    let is_boundary = (inf_vi >= 0);

    let bot_p = array<vec3<f32>, 4>(
        points[bot_tet.x].xyz,
        points[bot_tet.y].xyz,
        points[bot_tet.z].xyz,
        points[bot_tet.w].xyz,
    );

    // For non-boundary tets: compute tet orientation for orientation-agnostic insphere check.
    // For boundary tets: use tet_orient = 1 so the violation check (side * 1) > 0
    // directly maps CUDA's sphToSide semantics (violation when raw orient3d > 0).
    var tet_orient = 1;
    if !is_boundary {
        tet_orient = orient3d_with_sos(
            bot_p[0], bot_p[1], bot_p[2], bot_p[3],
            bot_tet.x, bot_tet.y, bot_tet.z, bot_tet.w
        );
    }

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

        var side: i32;
        if is_boundary {
            // CUDA boundary insphere fallback (KerPredWrapper.h:818-828):
            // orient3d(pt[ord[0]], pt[ord[2]], pt[ord[1]], ptVert) with SoS
            // sphToSide(ortToOrient(det)): violation when raw orient3d > 0
            // With tet_orient = 1: (side * 1) > 0 triggers violation when side > 0
            let inf_vi_u = u32(inf_vi);
            let ord = TET_VI_AS_SEEN_FROM[inf_vi_u];
            side = orient3d_with_sos(
                bot_p[ord.x], bot_p[ord.z], bot_p[ord.y], top_p,
                tet_vertex_val(bot_tet, ord.x), tet_vertex_val(bot_tet, ord.z),
                tet_vertex_val(bot_tet, ord.y), top_vert
            );
        } else {
            side = insphere_with_sos(
                bot_p[0], bot_p[1], bot_p[2], bot_p[3], top_p,
                bot_tet.x, bot_tet.y, bot_tet.z, bot_tet.w, top_vert
            );
        }

        // Violation check: (side * tet_orient) > 0
        // Non-boundary: tet_orient = orient3d sign; violation when insphere and orient agree
        // Boundary: tet_orient = 1; violation when orient3d(ord[0],ord[2],ord[1],top) > 0
        if (side * tet_orient) <= 0 {
            check_vi_local = check_vi_local >> 4u;
            continue; // No insphere failure
        }

        // We have insphere failure - set OPP_SPHERE_FAIL flag (CUDA: KerPredicates.cu:584)
        atomicAdd(&counters[3], 1u); // DEBUG: count sphere failures
        let old_opp = atomicLoad(&tet_opp[bot_ti_u * 4u + bot_vi]);
        atomicOr(&tet_opp[bot_ti_u * 4u + bot_vi], OPP_SPHERE_FAIL);

        if bot_cor_ord_vi < 3u {
            // 3-2 flip confirmed
            atomicAdd(&counters[4], 1u); // DEBUG: count 3-2 proposals
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
            // Direct indexing (now supported in wgpu 28+)
            let ord_vi_elem = bot_ord_vi[i];
            let fv = get_tet_vi_as_seen_from(ord_vi_elem);

            let p0 = bot_p[fv.x];
            let p1 = bot_p[fv.y];
            let p2 = bot_p[fv.z];

            // Extract vertex indices for SoS
            let va = bot_tet[fv.x];
            let vb = bot_tet[fv.y];
            let vc = bot_tet[fv.z];

            // Use exact orient3d with SoS
            var ort = orient3d_with_sos(p0, p1, p2, top_p, va, vb, vc, top_vert);
            // CUDA: doOrient3DSoS flips when v0/v1/v2 is infinity (KerPredWrapper.h:260-261)
            if va >= inf_idx || vb >= inf_idx || vc >= inf_idx {
                ort = -ort;
            }

            // gDel3D convention: orient3d < 0 for correct orientation.
            // CUDA: OrientPos (raw det < 0) required. Our -1 = raw det < 0.
            if ort != -1 {
                has_flip = false;
                atomicAdd(&counters[6], 1u); // DEBUG: count 2-3 orient failures
                break; // Cannot do 2-3 flip
            }
        }

        if has_flip {
            // 2-3 flip possible!
            atomicAdd(&counters[5], 1u); // DEBUG: count 2-3 proposals
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
