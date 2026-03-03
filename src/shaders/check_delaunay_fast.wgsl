// Port of kerCheckDelaunayFast from gDel3D/GPU/KerPredicates.cu (lines 391-637)
// Checks active tets for Delaunay violations and votes for flips
//
// For each active tet, checks all 4 faces:
// 1. Loads neighbor opposite vertices
// 2. Checks for 3-2 flip configuration (3 tets share an edge)
// 3. Does insphere test to detect violations
// 4. Checks orient3d for 2-3 flip feasibility
// 5. Votes for the flip by writing to vote arrays

@group(0) @binding(0) var<storage, read_write> act_tet_vec: array<i32>;
@group(0) @binding(1) var<storage, read> tets: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> tet_opp: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> tet_info: array<u32>;
@group(0) @binding(4) var<storage, read_write> tet_vote_arr: array<i32>;
@group(0) @binding(5) var<storage, read_write> vote_arr: array<i32>;
@group(0) @binding(6) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(8) var<uniform> params: vec4<u32>; // x = act_tet_num, y = vote_offset

const TET_ALIVE: u32 = 1u;
const TET_CHANGED: u32 = 2u;
const OPP_SPECIAL: u32 = 8u;
const OPP_SPHERE_FAIL: u32 = 16u;

const COUNTER_FLIP: u32 = 2u;

// TetViAsSeenFrom[vi] gives the 3 other vertices in counter-clockwise order when seen from vi
const TET_VI_AS_SEEN_FROM: array<vec3<u32>, 4> = array<vec3<u32>, 4>(
    vec3<u32>(1u, 3u, 2u), // seen from vertex 0
    vec3<u32>(0u, 2u, 3u), // seen from vertex 1
    vec3<u32>(0u, 3u, 1u), // seen from vertex 2
    vec3<u32>(0u, 1u, 2u), // seen from vertex 3
);

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

fn encode_opp(tet_idx: u32, vi: u32, special: bool, sphere_fail: bool) -> u32 {
    var result = (tet_idx << 5u) | vi;
    if special {
        result = result | OPP_SPECIAL;
    }
    if sphere_fail {
        result = result | OPP_SPHERE_FAIL;
    }
    return result;
}

fn make_flip(bot_vi: u32, bot_cor_ord_vi: u32) -> u32 {
    return (bot_cor_ord_vi << 2u) | bot_vi;
}

fn make_vote_val(bot_ti: u32, flip_info: u32) -> i32 {
    return i32((bot_ti << 8u) | flip_info);
}

fn vote_for_flip23(vote_offset: u32, bot_ti: u32, top_ti: u32) {
    let vote_val = i32(bot_ti);
    tet_vote_arr[vote_offset + bot_ti] = vote_val;
    tet_vote_arr[vote_offset + top_ti] = vote_val;
}

fn vote_for_flip32(vote_offset: u32, bot_ti: u32, top_ti: u32, bot_opp_ti: u32) {
    let vote_val = i32(bot_ti);
    tet_vote_arr[vote_offset + bot_ti] = vote_val;
    tet_vote_arr[vote_offset + top_ti] = vote_val;
    tet_vote_arr[vote_offset + bot_opp_ti] = vote_val;
}

fn orient3d_fast(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> i32 {
    let ad = a - d;
    let bd = b - d;
    let cd = c - d;
    let det = ad.x * (bd.y * cd.z - bd.z * cd.y)
            + bd.x * (cd.y * ad.z - cd.z * ad.y)
            + cd.x * (ad.y * bd.z - ad.z * bd.y);
    if det > 0.0 {
        return 1; // OrientPos
    } else if det < 0.0 {
        return -1; // OrientNeg
    }
    return 0; // OrientZero
}

fn insphere_fast(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>, e: vec3<f32>) -> i32 {
    let ae = a - e;
    let be = b - e;
    let ce = c - e;
    let de = d - e;

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

    let alift = ae.x * ae.x + ae.y * ae.y + ae.z * ae.z;
    let blift = be.x * be.x + be.y * be.y + be.z * be.z;
    let clift = ce.x * ce.x + ce.y * ce.y + ce.z * ce.z;
    let dlift = de.x * de.x + de.y * de.y + de.z * de.z;

    let det = (dlift * abc - clift * dab) + (blift * cda - alift * bcd);

    if det > 0.0 {
        return 1; // SideIn (inside sphere - violation!)
    } else if det < 0.0 {
        return -1; // SideOut
    }
    return 0; // SideZero
}

@compute @workgroup_size(64)
fn check_delaunay_fast(
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
            top_vert = i32(tet[top_vi]);

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
            let side_vert = opp_vert[bot_ord_vi[i]];

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

        let side = insphere_fast(bot_p[0], bot_p[1], bot_p[2], bot_p[3], top_p);

        if side == 0 {
            // Mark as special for exact mode
            bot_opp[bot_vi] = bot_opp[bot_vi] | OPP_SPECIAL;
        }

        if side != 1 {
            check_vi_local = check_vi_local >> 4u;
            continue; // No insphere failure
        }

        // We have insphere failure
        bot_opp[bot_vi] = bot_opp[bot_vi] | OPP_SPHERE_FAIL;

        if bot_cor_ord_vi < 3u {
            // 3-2 flip confirmed
            let flip_info = make_flip(bot_vi, bot_cor_ord_vi);
            vote_arr[idx] = make_vote_val(bot_ti_u, flip_info);

            let bot_ord_vi = TET_VI_AS_SEEN_FROM[bot_vi];
            let bot_cor_vi = bot_ord_vi[bot_cor_ord_vi];
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
            let fv = TET_VI_AS_SEEN_FROM[bot_ord_vi[i]];

            let ort = orient3d_fast(
                bot_p[fv.x], bot_p[fv.y], bot_p[fv.z], top_p
            );

            if ort == 0 {
                // Mark as special for exact mode
                bot_opp[bot_vi] = bot_opp[bot_vi] | OPP_SPECIAL;
            }

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

    // Store updated opp info
    atomicStore(&tet_opp[bot_ti_u * 4u + 0u], bot_opp[0]);
    atomicStore(&tet_opp[bot_ti_u * 4u + 1u], bot_opp[1]);
    atomicStore(&tet_opp[bot_ti_u * 4u + 2u], bot_opp[2]);
    atomicStore(&tet_opp[bot_ti_u * 4u + 3u], bot_opp[3]);

    // Reset flip counter (done by thread 0 of block 0)
    if gid.x == 0u {
        atomicStore(&counters[COUNTER_FLIP], 0u);
    }
}
