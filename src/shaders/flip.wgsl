// Kernels: Delaunay flipping (2-3 and 3-2 flips) with full execution.
//
// After point insertion, check each new tet's external faces for Delaunay
// violation (insphere test). If violated, perform a bistellar flip.
//
// 2-3 flip: Two tets sharing a face → three tets sharing an edge.
//   T_abc_d + T_abc_e → T_ab_de + T_bc_de + T_ca_de (net +1 tet)
//
// 3-2 flip: Three tets sharing an edge → two tets sharing a face.
//   T_ab_de + T_bc_de + T_ca_de → T_abc_d + T_abc_e (net -1 tet)
//
// Uses bilateral CAS locking to prevent concurrent modification of the
// same tets by different threads.
//
// CUDA Reference: KerDivision.cu:357-559 (kerFlip)

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> tets: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> tet_opp: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> tet_info: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> free_arr: array<u32>;
@group(0) @binding(5) var<storage, read_write> vert_free_arr: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> flip_queue: array<u32>;
@group(0) @binding(8) var<storage, read_write> flip_queue_next: array<u32>;
@group(0) @binding(9) var<storage, read_write> flip_count: array<atomic<u32>>; // [0] = next queue size
@group(0) @binding(10) var<uniform> params: vec4<u32>; // x = queue_size, y = inf_idx, z = use_exact, w = org_flip_num
@group(0) @binding(11) var<storage, read> block_owner: array<u32>; // Pre-computed block ownership
@group(0) @binding(12) var<storage, read_write> tet_vote: array<atomic<i32>>; // Flip voting
@group(0) @binding(13) var<storage, read_write> tet_msg_arr: array<vec2<i32>>;
@group(0) @binding(14) var<storage, read_write> flip_arr: array<vec4<i32>>; // FlipItem = 2 x vec4<i32>
@group(0) @binding(15) var<storage, read_write> encoded_face_vi_arr: array<i32>;

const INVALID: u32 = 0xFFFFFFFFu;
const TET_ALIVE: u32 = 1u;
const TET_CHANGED: u32 = 2u;
const TET_CHECKED: u32 = 4u;
const TET_LOCKED: u32 = 8u;
const OPP_INTERNAL: u32 = 4u; // Bit 2: marks adjacency between tets created by same flip
const COUNTER_FREE: u32 = 0u;
const COUNTER_ACTIVE: u32 = 1u;
const MEAN_VERTEX_DEGREE: u32 = 8u;

// Flip32NewFaceVi[3][2] from GPUDecl.h:148-152
// Maps old tet index (0=bot, 1=top, 2=side) to new face indices
const FLIP32_NEW_FACE_VI: array<vec2<u32>, 3> = array<vec2<u32>, 3>(
    vec2<u32>(2u, 1u),  // newTetIdx[0]'s faces
    vec2<u32>(1u, 2u),  // newTetIdx[1]'s faces
    vec2<u32>(0u, 0u),  // newTetIdx[2]'s faces
);

// TetViAsSeenFrom[vi] - 3 other vertices in CCW order when seen from vi
const TET_VI_AS_SEEN_FROM: array<vec3<u32>, 4> = array<vec3<u32>, 4>(
    vec3<u32>(1u, 3u, 2u),
    vec3<u32>(0u, 2u, 3u),
    vec3<u32>(0u, 3u, 1u),
    vec3<u32>(0u, 1u, 2u),
);

// Direct array indexing (now supported in wgpu 28+)
fn get_tet_vi_as_seen_from(vi: u32) -> vec3<u32> {
    return TET_VI_AS_SEEN_FROM[vi];
}

// TetNextViAsSeenFrom[4][4] - navigation around tet (CommonTypes.h:132-137)
// Returns -1 for self-index, otherwise index position
// Note: WGSL doesn't support -1 in u32, so we use a function instead
fn tet_next_vi(from_vi: u32, target_vi: u32) -> u32 {
    if from_vi == 0u {
        if target_vi == 0u { return 999u; }  // Invalid
        if target_vi == 1u { return 0u; }
        if target_vi == 2u { return 2u; }
        return 1u;  // target_vi == 3
    } else if from_vi == 1u {
        if target_vi == 1u { return 999u; }  // Invalid
        if target_vi == 0u { return 0u; }
        if target_vi == 2u { return 1u; }
        return 2u;  // target_vi == 3
    } else if from_vi == 2u {
        if target_vi == 2u { return 999u; }  // Invalid
        if target_vi == 0u { return 0u; }
        if target_vi == 1u { return 2u; }
        return 1u;  // target_vi == 3
    } else {  // from_vi == 3
        if target_vi == 3u { return 999u; }  // Invalid
        if target_vi == 0u { return 0u; }
        if target_vi == 1u { return 1u; }
        return 2u;  // target_vi == 2
    }
}

fn encode_opp(tet_idx: u32, face: u32) -> u32 {
    return (tet_idx << 5u) | (face & 3u);
}

fn encode_opp_internal(tet_idx: u32, face: u32) -> u32 {
    return (tet_idx << 5u) | OPP_INTERNAL | (face & 3u);
}

fn decode_opp_tet(packed: u32) -> u32 {
    return packed >> 5u;
}

fn decode_opp_face(packed: u32) -> u32 {
    return packed & 3u;
}

// --- Flat atomic opp accessors ---

fn get_opp(tet_idx: u32, face: u32) -> u32 {
    return atomicLoad(&tet_opp[tet_idx * 4u + face]);
}

fn set_opp_at(tet_idx: u32, face: u32, val: u32) {
    atomicStore(&tet_opp[tet_idx * 4u + face], val);
}

fn tet_vertex(tet: vec4<u32>, i: u32) -> u32 {
    switch i {
        case 0u: { return tet.x; }
        case 1u: { return tet.y; }
        case 2u: { return tet.z; }
        default: { return tet.w; }
    }
}

fn set_tet_entry(tet: vec4<u32>, i: u32, val: u32) -> vec4<u32> {
    var r = tet;
    switch i {
        case 0u: { r.x = val; }
        case 1u: { r.y = val; }
        case 2u: { r.z = val; }
        default: { r.w = val; }
    }
    return r;
}

fn insphere_simple(
    a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>, e: vec3<f32>
) -> f32 {
    let ae = a - e; let be = b - e; let ce = c - e; let de = d - e;

    let ab = ae.x * be.y - be.x * ae.y;
    let bc = be.x * ce.y - ce.x * be.y;
    let cd = ce.x * de.y - de.x * ce.y;
    let da = de.x * ae.y - ae.x * de.y;
    let ac = ae.x * ce.y - ce.x * ae.y;
    let bd = be.x * de.y - de.x * be.y;

    let abc = ae.z * bc - be.z * ac + ce.z * ab;
    let bcd = be.z * cd - ce.z * bd + de.z * bc;
    let cda = ce.z * da + de.z * ac + ae.z * cd;
    let dab = de.z * ab + ae.z * bd + be.z * da;

    let al = dot(ae, ae);
    let bl = dot(be, be);
    let cl = dot(ce, ce);
    let dl = dot(de, de);

    return (dl * abc - cl * dab) + (bl * cda - al * bcd);
}

fn orient3d_simple(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> f32 {
    let ad = a - d; let bd = b - d; let cd = c - d;
    return ad.x * (bd.y * cd.z - bd.z * cd.y)
         + bd.x * (cd.y * ad.z - cd.z * ad.y)
         + cd.x * (ad.y * bd.z - ad.z * bd.y);
}

// --- Double-double (DD) arithmetic for exact predicates ---
struct DD { hi: f32, lo: f32, }

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

// Exact insphere using DD arithmetic (returns sign: +1 inside, -1 outside, 0 on sphere)
fn insphere_exact(
    a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>, e: vec3<f32>
) -> i32 {
    // Try fast f32 first with error bound check
    let ae = a - e; let be = b - e; let ce = c - e; let de = d - e;
    let ab = ae.x * be.y - be.x * ae.y;
    let bc = be.x * ce.y - ce.x * be.y;
    let cd = ce.x * de.y - de.x * ce.y;
    let da = de.x * ae.y - ae.x * de.y;
    let ac = ae.x * ce.y - ce.x * ae.y;
    let bd = be.x * de.y - de.x * be.y;

    let abc = ae.z * bc - be.z * ac + ce.z * ab;
    let bcd = be.z * cd - ce.z * bd + de.z * bc;
    let cda = ce.z * da + de.z * ac + ae.z * cd;
    let dab = de.z * ab + ae.z * bd + be.z * da;

    let al = dot(ae, ae); let bl = dot(be, be);
    let cl = dot(ce, ce); let dl = dot(de, de);

    let det = (dl * abc - cl * dab) + (bl * cda - al * bcd);
    let permanent = abs(dl * abc) + abs(cl * dab) + abs(bl * cda) + abs(al * bcd);
    let err_bound = 1.5e-5 * permanent;

    if abs(det) > err_bound {
        if det > 0.0 { return 1; }
        return -1;
    }

    // Fall back to DD arithmetic
    let aex = dd_from_f32(a.x - e.x); let aey = dd_from_f32(a.y - e.y); let aez = dd_from_f32(a.z - e.z);
    let bex = dd_from_f32(b.x - e.x); let bey = dd_from_f32(b.y - e.y); let bez = dd_from_f32(b.z - e.z);
    let cex = dd_from_f32(c.x - e.x); let cey = dd_from_f32(c.y - e.y); let cez = dd_from_f32(c.z - e.z);
    let dex = dd_from_f32(d.x - e.x); let dey = dd_from_f32(d.y - e.y); let dez = dd_from_f32(d.z - e.z);

    let ab_dd = dd_sub(dd_mul(aex, bey), dd_mul(bex, aey));
    let bc_dd = dd_sub(dd_mul(bex, cey), dd_mul(cex, bey));
    let cd_dd = dd_sub(dd_mul(cex, dey), dd_mul(dex, cey));
    let da_dd = dd_sub(dd_mul(dex, aey), dd_mul(aex, dey));
    let ac_dd = dd_sub(dd_mul(aex, cey), dd_mul(cex, aey));
    let bd_dd = dd_sub(dd_mul(bex, dey), dd_mul(dex, bey));

    let abc_dd = dd_add(dd_sub(dd_mul(aez, bc_dd), dd_mul(bez, ac_dd)), dd_mul(cez, ab_dd));
    let bcd_dd = dd_add(dd_sub(dd_mul(bez, cd_dd), dd_mul(cez, bd_dd)), dd_mul(dez, bc_dd));
    let cda_dd = dd_add(dd_add(dd_mul(cez, da_dd), dd_mul(dez, ac_dd)), dd_mul(aez, cd_dd));
    let dab_dd = dd_add(dd_add(dd_mul(dez, ab_dd), dd_mul(aez, bd_dd)), dd_mul(bez, da_dd));

    let alift = dd_add(dd_add(dd_mul(aex, aex), dd_mul(aey, aey)), dd_mul(aez, aez));
    let blift = dd_add(dd_add(dd_mul(bex, bex), dd_mul(bey, bey)), dd_mul(bez, bez));
    let clift = dd_add(dd_add(dd_mul(cex, cex), dd_mul(cey, cey)), dd_mul(cez, cez));
    let dlift = dd_add(dd_add(dd_mul(dex, dex), dd_mul(dey, dey)), dd_mul(dez, dez));

    let det_dd = dd_add(
        dd_sub(dd_mul(dlift, abc_dd), dd_mul(clift, dab_dd)),
        dd_sub(dd_mul(blift, cda_dd), dd_mul(alift, bcd_dd))
    );
    return dd_sign(det_dd);
}

// Exact orient3d using DD arithmetic
fn orient3d_exact(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> i32 {
    // Try fast f32 first with error bound
    let ad = a - d; let bd = b - d; let cd = c - d;
    let det = ad.x * (bd.y * cd.z - bd.z * cd.y)
            + bd.x * (cd.y * ad.z - cd.z * ad.y)
            + cd.x * (ad.y * bd.z - ad.z * bd.y);
    let permanent = abs(ad.x) * (abs(bd.y * cd.z) + abs(bd.z * cd.y))
                  + abs(bd.x) * (abs(cd.y * ad.z) + abs(cd.z * ad.y))
                  + abs(cd.x) * (abs(ad.y * bd.z) + abs(ad.z * bd.y));
    let err_bound = 7.77e-7 * permanent;

    if abs(det) > err_bound {
        if det > 0.0 { return 1; }
        return -1;
    }

    // Fall back to DD
    let adx = dd_from_f32(a.x - d.x); let ady = dd_from_f32(a.y - d.y); let adz = dd_from_f32(a.z - d.z);
    let bdx = dd_from_f32(b.x - d.x); let bdy = dd_from_f32(b.y - d.y); let bdz = dd_from_f32(b.z - d.z);
    let cdx = dd_from_f32(c.x - d.x); let cdy = dd_from_f32(c.y - d.y); let cdz = dd_from_f32(c.z - d.z);

    let t1 = dd_sub(dd_mul(bdy, cdz), dd_mul(bdz, cdy));
    let t2 = dd_sub(dd_mul(cdy, adz), dd_mul(cdz, ady));
    let t3 = dd_sub(dd_mul(ady, bdz), dd_mul(adz, bdy));

    let det_dd = dd_add(dd_add(dd_mul(adx, t1), dd_mul(bdx, t2)), dd_mul(cdx, t3));
    return dd_sign(det_dd);
}

// Try to CAS-lock a tet. Returns true on success.
fn try_lock(tet_idx: u32) -> bool {
    let cur = atomicLoad(&tet_info[tet_idx]);
    if (cur & TET_ALIVE) == 0u || (cur & TET_LOCKED) != 0u {
        return false;
    }
    let result = atomicCompareExchangeWeak(&tet_info[tet_idx], cur, cur | TET_LOCKED);
    return result.exchanged;
}

// Unlock a tet (clear the lock bit, keep other flags).
fn unlock(tet_idx: u32) {
    let cur = atomicLoad(&tet_info[tet_idx]);
    atomicStore(&tet_info[tet_idx], cur & ~TET_LOCKED);
}

// Find which local index of `tet` contains vertex `v`. Returns 4 if not found.
fn find_local(tet: vec4<u32>, v: u32) -> u32 {
    if tet.x == v { return 0u; }
    if tet.y == v { return 1u; }
    if tet.z == v { return 2u; }
    if tet.w == v { return 3u; }
    return 4u;
}

// setTetIdxVi: Encode old face position → new tet mapping
// CUDA: KerDivision.cu:337-341
fn set_tet_idx_vi(output: i32, old_vi: u32, ni: u32, new_vi: u32) -> i32 {
    return output - i32(0xFu << (old_vi * 4u)) + i32(((ni << 2u) + new_vi) << (old_vi * 4u));
}

// Access vec3 component by runtime index (avoid variable indexing into const arrays)
fn vec3_at(v: vec3<u32>, i: u32) -> u32 {
    if i == 0u { return v.x; }
    if i == 1u { return v.y; }
    return v.z;
}

// Donate tet to vertex owner's free list (for 3-2 flips)
// CUDA: KerDivision.cu:495-500
// Calculate block ownership and donate to that vertex's pool (or inf_idx if beyond inserted vertices)
fn donate_tet(tet_idx: u32) {
    // Calculate which block this tet belongs to (CUDA line 496)
    let blk_idx = tet_idx / MEAN_VERTEX_DEGREE;

    // Get the owner vertex (CUDA line 497: uses insVertVec._arr[insIdx] or infIdx)
    // block_owner already does this mapping: maps block index to vertex owner
    let owner_vertex = block_owner[blk_idx];

    // Atomically allocate slot in owner's free list (CUDA line 498)
    let free_slot = atomicAdd(&vert_free_arr[owner_vertex], 1u);

    // Store donated tet in owner's free list (CUDA line 500)
    free_arr[owner_vertex * MEAN_VERTEX_DEGREE + free_slot] = tet_idx;

    // Mark tet as dead
    atomicAnd(&tet_info[tet_idx], ~TET_ALIVE);
}

// Allocate slot for 2-3 flip using per-vertex free lists (CUDA: kerAllocateFlip23Slot, lines 888-953)
// Tries vertices of bottom tet first, then falls back to inf_idx pool.
//
// Allocate 1 tet slot from the global tet pool.
// Uses vert_free_arr[0] as a global atomic counter (next free tet index).
//
// This replaces the per-vertex free list approach which had a race condition:
// concurrent atomicAdd (donation) and atomicSub (allocation) on the same
// per-vertex counter could overflow the 8-slot block boundary, causing
// allocations to read tet indices from adjacent blocks (potentially alive tets).
fn allocate_flip23_slot() -> u32 {
    return atomicAdd(&vert_free_arr[0], 1u);
}

// Phase 1: Vote for flip via atomicMin to resolve conflicts between concurrent flips.
// CUDA Reference: KerPredicates.cu:326-356 (voteForFlip23/voteForFlip32)
// Each thread detects the first Delaunay violation and votes atomicMin on all involved tets.
// The lowest tet index wins, preventing topological conflicts.
@compute @workgroup_size(256)
fn flip_vote(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let queue_size = params.x;
    if idx >= queue_size { return; }

    let tet_a = flip_queue[idx];
    let info_a = atomicLoad(&tet_info[tet_a]);
    if (info_a & TET_ALIVE) == 0u { return; }

    let tet_a_data = tets[tet_a];

    for (var face_a = 0u; face_a < 4u; face_a++) {
        let opp_packed = get_opp(tet_a, face_a);
        if opp_packed == INVALID { continue; }
        // CUDA: KerPredicates.cu:443 — skip internal faces (already locally Delaunay)
        // Internal faces are between tets created by the same flip (siblings).
        if (opp_packed & OPP_INTERNAL) != 0u { continue; }

        let tet_b = decode_opp_tet(opp_packed);
        let face_b = decode_opp_face(opp_packed);

        let info_b = atomicLoad(&tet_info[tet_b]);
        if (info_b & TET_ALIVE) == 0u { continue; }

        let tet_b_data = tets[tet_b];
        let va = tet_vertex(tet_a_data, face_a);
        let vb = tet_vertex(tet_b_data, face_b);

        let inf_idx = params.y;
        if (va >= inf_idx) || (vb >= inf_idx) { continue; }

        var face_v: array<u32, 3>;
        var si = 0u;
        for (var i = 0u; i < 4u; i++) {
            let v = tet_vertex(tet_a_data, i);
            if v != va { face_v[si] = v; si++; }
        }

        if (face_v[0] >= inf_idx) || (face_v[1] >= inf_idx) || (face_v[2] >= inf_idx) { continue; }

        let pa = points[va].xyz;
        let pb = points[vb].xyz;
        let p0 = points[face_v[0]].xyz;
        let p1 = points[face_v[1]].xyz;
        let p2 = points[face_v[2]].xyz;

        // Insphere test — orientation-agnostic (handles both gDel3D and Shewchuk tets).
        // Violation = point inside circumsphere = insphere * orient > 0.
        let use_exact = params.z != 0u;
        var violation = false;
        let t0 = points[tet_a_data.x].xyz;
        let t1 = points[tet_a_data.y].xyz;
        let t2 = points[tet_a_data.z].xyz;
        let t3 = points[tet_a_data.w].xyz;

        if use_exact {
            let insph = insphere_exact(t0, t1, t2, t3, pb);
            let orient = orient3d_exact(t0, t1, t2, t3);
            violation = (insph * orient) > 0;
        } else {
            let insph_f = insphere_simple(t0, t1, t2, t3, pb);
            let ad = t0 - t3; let bd = t1 - t3; let cd = t2 - t3;
            let orient_f = ad.x * (bd.y * cd.z - bd.z * cd.y)
                         + bd.x * (cd.y * ad.z - cd.z * ad.y)
                         + cd.x * (ad.y * bd.z - ad.z * bd.y);
            // Use sign comparison to avoid f32 underflow when both values are tiny.
            // (insph_f * orient_f) can underflow to 0.0 near degenerate configs.
            violation = (insph_f > 0.0 && orient_f > 0.0) || (insph_f < 0.0 && orient_f < 0.0);
        }

        if !violation { continue; }

        // Detect 3-2 flip configuration (same logic as flip_check)
        let cor_vi = TET_VI_AS_SEEN_FROM[face_a];
        var is_flip32 = false;
        var bot_cor_ord_vi = 3u;
        for (var i = 0u; i < 3u; i++) {
            let edge_vi = select(cor_vi.x, select(cor_vi.y, cor_vi.z, i == 2u), i >= 1u);
            let opp_edge = get_opp(tet_a, edge_vi);
            if opp_edge == INVALID { continue; }
            // Skip internal faces in 3-2 detection — prevents detecting the reverse
            // configuration through sibling internal faces (2-3 ↔ 3-2 oscillation).
            // CUDA: kerCheckDelaunay also skips internal faces in the 3-tet detection.
            if (opp_edge & OPP_INTERNAL) != 0u { continue; }
            let edge_nei_tet = decode_opp_tet(opp_edge);
            let edge_nei_face = decode_opp_face(opp_edge);
            let edge_nei_data = tets[edge_nei_tet];
            let edge_nei_opp_v = tet_vertex(edge_nei_data, edge_nei_face);
            if edge_nei_opp_v == vb {
                is_flip32 = true;
                bot_cor_ord_vi = i;
                break;
            }
        }

        // Check 2-3 viability before voting.
        // CUDA: KerPredicates.cu:562-583 — check orient3d of each sub-face with topVert.
        // topVert must be on the OrientPos side (Shewchuk orient3d < 0) of ALL 3 sub-faces
        // adjacent to the violating face. This ensures new tets preserve orient3d < 0.
        if !is_flip32 {
            var flip23_ok = true;
            for (var j = 0u; j < 3u; j++) {
                let side_vi = select(cor_vi.x, select(cor_vi.y, cor_vi.z, j == 2u), j >= 1u);
                let fv = get_tet_vi_as_seen_from(side_vi);
                let fp0 = points[tet_vertex(tet_a_data, fv.x)].xyz;
                let fp1 = points[tet_vertex(tet_a_data, fv.y)].xyz;
                let fp2 = points[tet_vertex(tet_a_data, fv.z)].xyz;
                if orient3d_exact(fp0, fp1, fp2, pb) >= 0 {
                    flip23_ok = false;
                    break;
                }
            }
            if !flip23_ok { continue; }
        }

        // Vote via atomicMin (CUDA: voteForFlip23/voteForFlip32)
        // The lowest tet index wins, preventing topological conflicts.
        let vote_val = i32(tet_a);
        atomicMin(&tet_vote[tet_a], vote_val);
        atomicMin(&tet_vote[tet_b], vote_val);

        if is_flip32 {
            let bot_cor_vi_idx = select(cor_vi.x, select(cor_vi.y, cor_vi.z, bot_cor_ord_vi == 2u), bot_cor_ord_vi >= 1u);
            let side_opp = get_opp(tet_a, bot_cor_vi_idx);
            if side_opp != INVALID {
                let tet_c = decode_opp_tet(side_opp);
                atomicMin(&tet_vote[tet_c], vote_val);
            }
        }

        return; // One vote per thread
    }
}

// Phase 2: Execute flips for tets validated by check_delaunay + mark_rejected_flips.
// Queue entries are packed: (tet_idx << 4) | flip_info
// flip_info = (bot_cor_ord_vi << 2) | bot_vi
// - bot_vi: which face of tet_a is violated
// - bot_cor_ord_vi: 0-2 = 3-2 flip (which neighbor shares edge), 3 = 2-3 flip
@compute @workgroup_size(256)
fn flip_check(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let queue_size = params.x;

    if idx >= queue_size {
        return;
    }

    // Decode packed queue entry
    let packed = flip_queue[idx];
    let tet_a = packed >> 4u;
    let flip_info = packed & 0xFu;
    let face_a = flip_info & 3u;
    var bot_cor_ord_vi = (flip_info >> 2u) & 3u;
    var is_flip32 = bot_cor_ord_vi < 3u;

    // DEBUG: counters[4]=invalid_opp, [5]=dead_tet_b, [6]=lock_fail, [7]=revalidation_fail

    // Skip dead tets
    let info_a = atomicLoad(&tet_info[tet_a]);
    if (info_a & TET_ALIVE) == 0u {
        // dead tet_a (counted in lock_fail slot for simplicity)
        atomicAdd(&counters[6], 1u);
        return;
    }

    let tet_a_data = tets[tet_a];
    let inf_idx = params.y;

    // Read adjacency for the specific violated face
    let opp_packed = get_opp(tet_a, face_a);
    if opp_packed == INVALID {
        atomicAdd(&counters[4], 1u); // INVALID opp (reuse dead_a slot)
        return;
    }

    let tet_b = decode_opp_tet(opp_packed);
    let face_b = decode_opp_face(opp_packed);

    let info_b = atomicLoad(&tet_info[tet_b]);
    if (info_b & TET_ALIVE) == 0u {
        atomicAdd(&counters[5], 1u); // dead tet_b
        return;
    }

    let tet_b_data = tets[tet_b];
    let va = tet_vertex(tet_a_data, face_a);
    let vb = tet_vertex(tet_b_data, face_b);

    let cor_vi = TET_VI_AS_SEEN_FROM[face_a];

    // === Try to lock both tets ===
    let first = min(tet_a, tet_b);
    let second = max(tet_a, tet_b);
    if !try_lock(first) {
        atomicAdd(&counters[6], 1u); // lock fail (first)
        return;
    }
    if !try_lock(second) {
        atomicAdd(&counters[6], 1u); // lock fail (second)
        unlock(first);
        return;
    }

    // === Re-validate after locking ===
    {
        let opp_recheck = get_opp(tet_a, face_a);
        if opp_recheck != opp_packed {
            atomicAdd(&counters[7], 1u); // revalidation fail (opp changed)
            unlock(second); unlock(first); return;
        }
        let back_opp = get_opp(tet_b, face_b);
        if decode_opp_tet(back_opp) != tet_a {
            atomicAdd(&counters[7], 1u); // revalidation fail (back_opp mismatch)
            unlock(second); unlock(first); return;
        }
    }

    if is_flip32 {
            // === Execute 3-2 flip ===
            // CUDA: KerDivision.cu:441-501
            // Three tets sharing edge (va,vb) → Two tets

            // Get the third tet (side tet) - CUDA line 444-446
            let bot_cor_vi_idx = select(cor_vi.x, select(cor_vi.y, cor_vi.z, bot_cor_ord_vi == 2u), bot_cor_ord_vi >= 1u);
            let side_opp = get_opp(tet_a, bot_cor_vi_idx);
            let tet_c = decode_opp_tet(side_opp);
            let side_cor_vi_0 = decode_opp_face(side_opp);

            // Try to lock tet_c (already hold first and second)
            // Note: tet_c might be between first and second in ordering,
            // but since we already hold both, just try tet_c directly.
            if !try_lock(tet_c) {
                atomicAdd(&counters[6], 1u); // lock fail (tet_c)
                unlock(second);
                unlock(first);
                return;
            }

            let tet_c_data = tets[tet_c];

            // Vertex identification (CUDA lines 448-466)
            // Get vertex indices for the 2 vertices from each old tet

            // Bot tet vertices (CUDA lines 449-452)
            let bot_a_vi = select(cor_vi.x, select(cor_vi.y, cor_vi.z, ((bot_cor_ord_vi + 1u) % 3u) == 2u), ((bot_cor_ord_vi + 1u) % 3u) >= 1u);
            let bot_b_vi = select(cor_vi.x, select(cor_vi.y, cor_vi.z, ((bot_cor_ord_vi + 2u) % 3u) == 2u), ((bot_cor_ord_vi + 2u) % 3u) >= 1u);
            let bot_a = tet_vertex(tet_a_data, bot_a_vi);
            let bot_b = tet_vertex(tet_a_data, bot_b_vi);

            // Corner vertex (shared by all 3 tets) - CUDA line 455
            let bot_cor = tet_vertex(tet_a_data, bot_cor_vi_idx);

            // Top tet vertices (CUDA lines 456-459)
            let cor_top_vi = TET_VI_AS_SEEN_FROM[face_b];
            let top_cor_vi = find_local(tet_b_data, bot_cor);
            let top_loc_vi = tet_next_vi(face_b, top_cor_vi);
            let top_a_vi = select(cor_top_vi.x, select(cor_top_vi.y, cor_top_vi.z, ((top_loc_vi + 2u) % 3u) == 2u), ((top_loc_vi + 2u) % 3u) >= 1u);
            let top_b_vi = select(cor_top_vi.x, select(cor_top_vi.y, cor_top_vi.z, ((top_loc_vi + 1u) % 3u) == 2u), ((top_loc_vi + 1u) % 3u) >= 1u);

            // Side tet vertices (CUDA lines 462-466)
            let top_opp_at_cor = get_opp(tet_b, top_cor_vi);
            let side_cor_vi_1 = decode_opp_face(top_opp_at_cor);
            let side_loc_vi = tet_next_vi(side_cor_vi_0, side_cor_vi_1);
            let side_ord_vi = TET_VI_AS_SEEN_FROM[side_cor_vi_0];
            let side_a_vi = select(side_ord_vi.x, select(side_ord_vi.y, side_ord_vi.z, ((side_loc_vi + 1u) % 3u) == 2u), ((side_loc_vi + 1u) % 3u) >= 1u);
            let side_b_vi = select(side_ord_vi.x, select(side_ord_vi.y, side_ord_vi.z, ((side_loc_vi + 2u) % 3u) == 2u), ((side_loc_vi + 2u) % 3u) >= 1u);

            // Old face vertex indices for metadata (CUDA lines 468-472)
            // oldFaceVi[ti][0..1] = the 2 face vertices from old tet ti
            let old_face_vi = array<vec2<u32>, 3>(
                vec2<u32>(bot_a_vi, bot_b_vi),    // Bot tet
                vec2<u32>(top_a_vi, top_b_vi),    // Top tet
                vec2<u32>(side_a_vi, side_b_vi),  // Side tet
            );

            // Compute newIdx and encodedFaceVi (CUDA lines 474-483)
            var new_idx_0: i32 = 0xFFFF;
            var new_idx_1: i32 = 0xFFFF;
            var new_idx_2: i32 = 0xFFFF;
            var encoded_face_vi: i32 = 0;

            // ti=0 (bot tet): face[0] → tet 1, face[1] → same slot (3)
            new_idx_0 = set_tet_idx_vi(new_idx_0, old_face_vi[0].x, 1u, FLIP32_NEW_FACE_VI[0].x);
            new_idx_0 = set_tet_idx_vi(new_idx_0, old_face_vi[0].y, 3u, FLIP32_NEW_FACE_VI[0].y);
            encoded_face_vi = (encoded_face_vi << 4) | i32((old_face_vi[0].x << 2u) | old_face_vi[0].y);

            // ti=1 (top tet): face[0] → same slot (3), face[1] → tet 0
            new_idx_1 = set_tet_idx_vi(new_idx_1, old_face_vi[1].x, 3u, FLIP32_NEW_FACE_VI[1].x);
            new_idx_1 = set_tet_idx_vi(new_idx_1, old_face_vi[1].y, 0u, FLIP32_NEW_FACE_VI[1].y);
            encoded_face_vi = (encoded_face_vi << 4) | i32((old_face_vi[1].x << 2u) | old_face_vi[1].y);

            // ti=2 (side tet): face[0] → tet 1, face[1] → tet 0
            new_idx_2 = set_tet_idx_vi(new_idx_2, old_face_vi[2].x, 1u, FLIP32_NEW_FACE_VI[2].x);
            new_idx_2 = set_tet_idx_vi(new_idx_2, old_face_vi[2].y, 0u, FLIP32_NEW_FACE_VI[2].y);
            encoded_face_vi = (encoded_face_vi << 4) | i32((old_face_vi[2].x << 2u) | old_face_vi[2].y);

            // Allocate flip metadata index
            let flip_idx = atomicAdd(&flip_count[1], 1u);
            let glob_flip_idx = params.w + flip_idx;

            // Write tet_msg_arr for side tet first (CUDA line 486)
            tet_msg_arr[tet_c] = vec2<i32>(new_idx_2, i32(glob_flip_idx));

            // Create new tets in CUDA order (KerDivision.cu:492-493)
            let bot_tet_v = va;
            let top_tet_v = vb;
            let c0 = vec4<u32>(bot_cor, top_tet_v, bot_tet_v, bot_a);
            let c1 = vec4<u32>(bot_cor, bot_tet_v, top_tet_v, bot_b);

            tets[tet_a] = c0;
            tets[tet_b] = c1;

            // Mark side tet as dead (CUDA lines 495-500)
            // NOTE: Unlike CUDA, we do NOT return tet_c to the free list here.
            // CUDA uses a separate kerAllocateFlip23Slot kernel that runs BEFORE kerFlip,
            // so 3-2 donated slots can't be immediately picked up by 2-3 flips in the
            // same dispatch. Our combined kernel would create a race: 3-2 frees slot X,
            // concurrent 2-3 allocates slot X → both flips reference the same tet slot
            // → update_flip_trace chain corruption → wrong vert_tet after relocate.
            // Just marking dead is safe; we pre-allocate enough free slots at startup.
            atomicAnd(&tet_info[tet_c], ~TET_ALIVE);

            // Write tet_msg_arr for bot and top (CUDA lines 504-505)
            tet_msg_arr[tet_a] = vec2<i32>(new_idx_0, i32(glob_flip_idx));
            tet_msg_arr[tet_b] = vec2<i32>(new_idx_1, i32(glob_flip_idx));

            // Write encodedFaceVi (CUDA line 512)
            // CRITICAL: Use glob_flip_idx for flip_arr/encoded_face_vi_arr writes
            // so update_flip_trace can read at global indices (CUDA writes at flipArr[globFlipIdx])
            encoded_face_vi_arr[glob_flip_idx] = encoded_face_vi;

            // Write FlipItem (CUDA lines 521-524)
            // Layout: [_v[0..3], _v[4], _t[0], _t[1], _t[2]]
            // _v[0..3] = c0 vertices, _v[4] = c1[3] = bot_b
            // _t[2] = -tet_c (negative = Flip32)
            flip_arr[glob_flip_idx * 2u] = vec4<i32>(i32(c0.x), i32(c0.y), i32(c0.z), i32(c0.w));
            // CUDA: _t[2] = makeNegative(sideTetIdx) = -(sideTetIdx + 2)
            // Must match update_flip_trace.wgsl's make_positive: -(v + 2)
            flip_arr[glob_flip_idx * 2u + 1u] = vec4<i32>(i32(bot_b), i32(tet_a), i32(tet_b), -(i32(tet_c) + 2));

            atomicStore(&tet_info[tet_a], TET_ALIVE | TET_CHANGED);
            atomicStore(&tet_info[tet_b], TET_ALIVE | TET_CHANGED);

            // Update active count (3 → 2 = net -1)
            atomicSub(&counters[COUNTER_ACTIVE], 1u);

            // Enqueue for next round
            let slot = atomicAdd(&flip_count[0], 2u);
            flip_queue_next[slot] = tet_a;
            flip_queue_next[slot + 1u] = tet_b;

            return;
        }

        // === Execute 2-3 flip ===
        // CUDA: KerDivision.cu:405-440
        // Two tets sharing a face → three tets sharing an edge.

        // Allocate 1 slot from per-vertex free lists (2 tets → 3 tets = net +1)
        let new_slot = allocate_flip23_slot();
        if new_slot == INVALID {
            unlock(second);
            unlock(first);
            return;
        }

        // Shared face vertices in TetViAsSeenFrom order (CUDA: corV)
        // cor_vi was already computed: TET_VI_AS_SEEN_FROM[face_a]
        let cor_v_0 = tet_vertex(tet_a_data, cor_vi.x);
        let cor_v_1 = tet_vertex(tet_a_data, cor_vi.y);
        let cor_v_2 = tet_vertex(tet_a_data, cor_vi.z);

        // Alignment of top tet's face ordering (CUDA line 408)
        let cor_top_vi = TET_VI_AS_SEEN_FROM[face_b];
        let top_vi0 = tet_next_vi(face_b, find_local(tet_b_data, cor_v_2));

        // oldFaceVi[ni][0..1] = old bot/top face indices (CUDA lines 410-414)
        let old_face_vi_23 = array<vec2<u32>, 3>(
            vec2<u32>(cor_vi.z, vec3_at(cor_top_vi, top_vi0)),
            vec2<u32>(cor_vi.x, vec3_at(cor_top_vi, (top_vi0 + 2u) % 3u)),
            vec2<u32>(cor_vi.y, vec3_at(cor_top_vi, (top_vi0 + 1u) % 3u)),
        );

        // Compute newIdx and encodedFaceVi (CUDA lines 417-426)
        var new_idx_0: i32 = 0xFFFF;
        var new_idx_1: i32 = 0xFFFF;
        var encoded_face_vi_23: i32 = 0;

        // ni=0: bot face stays (ni=3), top face → tet 0 face 3 (ni=0)
        new_idx_0 = set_tet_idx_vi(new_idx_0, old_face_vi_23[0].x, 3u, 0u);
        new_idx_1 = set_tet_idx_vi(new_idx_1, old_face_vi_23[0].y, 0u, 3u);
        encoded_face_vi_23 = (encoded_face_vi_23 << 4) | i32((old_face_vi_23[0].x << 2u) | old_face_vi_23[0].y);

        // ni=1: bot face → tet 1 face 0 (ni=1), top face stays (ni=3)
        new_idx_0 = set_tet_idx_vi(new_idx_0, old_face_vi_23[1].x, 1u, 0u);
        new_idx_1 = set_tet_idx_vi(new_idx_1, old_face_vi_23[1].y, 3u, 3u);
        encoded_face_vi_23 = (encoded_face_vi_23 << 4) | i32((old_face_vi_23[1].x << 2u) | old_face_vi_23[1].y);

        // ni=2: both → tet 2 (ni=2)
        new_idx_0 = set_tet_idx_vi(new_idx_0, old_face_vi_23[2].x, 2u, 0u);
        new_idx_1 = set_tet_idx_vi(new_idx_1, old_face_vi_23[2].y, 2u, 3u);
        encoded_face_vi_23 = (encoded_face_vi_23 << 4) | i32((old_face_vi_23[2].x << 2u) | old_face_vi_23[2].y);

        // Allocate flip metadata index
        let flip_idx = atomicAdd(&flip_count[1], 1u);
        let glob_flip_idx = params.w + flip_idx;

        // Create new tets in CUDA order (KerDivision.cu:432-436)
        // makeTet(topVert, corV[i], corV[(i+1)%3], botVert)
        let top_vert = vb;
        let bot_vert = va;
        let c0 = vec4<u32>(top_vert, cor_v_0, cor_v_1, bot_vert);
        let c1 = vec4<u32>(top_vert, cor_v_1, cor_v_2, bot_vert);
        let c2 = vec4<u32>(top_vert, cor_v_2, cor_v_0, bot_vert);

        tets[tet_a] = c0;
        tets[tet_b] = c1;
        tets[new_slot] = c2;

        // Write tetMsgArr (CUDA lines 504-505)
        tet_msg_arr[tet_a] = vec2<i32>(new_idx_0, i32(glob_flip_idx));
        tet_msg_arr[tet_b] = vec2<i32>(new_idx_1, i32(glob_flip_idx));

        // Write encodedFaceVi (CUDA line 512)
        // CRITICAL: Use glob_flip_idx for flip_arr/encoded_face_vi_arr writes
        // so update_flip_trace can read at global indices (CUDA writes at flipArr[globFlipIdx])
        encoded_face_vi_arr[glob_flip_idx] = encoded_face_vi_23;

        // Write FlipItem (CUDA lines 521-524)
        // Layout: [_v[0..3], _v[4], _t[0], _t[1], _t[2]]
        // _v[0..3] = c0 vertices, _v[4] = c1[2] = cor_v_2 (Flip23)
        // _t[2] = new_slot (positive = Flip23)
        flip_arr[glob_flip_idx * 2u] = vec4<i32>(i32(c0.x), i32(c0.y), i32(c0.z), i32(c0.w));
        flip_arr[glob_flip_idx * 2u + 1u] = vec4<i32>(i32(cor_v_2), i32(tet_a), i32(tet_b), i32(new_slot));

        // Mark new tets as alive
        atomicStore(&tet_info[tet_a], TET_ALIVE | TET_CHANGED);
        atomicStore(&tet_info[tet_b], TET_ALIVE | TET_CHANGED);
        atomicStore(&tet_info[new_slot], TET_ALIVE | TET_CHANGED);

        // Update active count (2 → 3 = net +1)
        atomicAdd(&counters[COUNTER_ACTIVE], 1u);

        // Enqueue all 3 tets for next flip round
        let slot = atomicAdd(&flip_count[0], 3u);
        flip_queue_next[slot] = tet_a;
        flip_queue_next[slot + 1u] = tet_b;
        flip_queue_next[slot + 2u] = new_slot;

        return;
}
