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
@group(0) @binding(10) var<uniform> params: vec4<u32>; // x = queue_size, y = inf_idx

const INVALID: u32 = 0xFFFFFFFFu;
const TET_ALIVE: u32 = 1u;
const TET_CHANGED: u32 = 2u;
const TET_CHECKED: u32 = 4u;
const TET_LOCKED: u32 = 8u;
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

// CRITICAL FIX: Helper function to avoid variable array indexing (causes SIGSEGV)
// Cannot use TET_VI_AS_SEEN_FROM[variable] - must use explicit branches
fn get_tet_vi_as_seen_from(vi: u32) -> vec3<u32> {
    if vi == 0u { return vec3<u32>(1u, 3u, 2u); }
    else if vi == 1u { return vec3<u32>(0u, 2u, 3u); }
    else if vi == 2u { return vec3<u32>(0u, 3u, 1u); }
    else { return vec3<u32>(0u, 1u, 2u); }
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

// Donate tet to infinity free list (for 3-2 flips)
// CUDA: KerDivision.cu:495-500
// Simplified: donate to inf_idx (params.y) instead of calculating ownership
fn donate_tet(tet_idx: u32) {
    let inf_idx = params.y;
    let free_slot = atomicAdd(&vert_free_arr[inf_idx], 1u);
    free_arr[inf_idx * MEAN_VERTEX_DEGREE + free_slot] = tet_idx;
    // Mark tet as dead
    atomicAnd(&tet_info[tet_idx], ~TET_ALIVE);
}

// Pop 1 free slot from the free stack.
fn pop_free_slot() -> u32 {
    let old_free = atomicSub(&counters[COUNTER_FREE], 1u);
    return free_arr[old_free - 1u];
}

// Check and perform 2-3 flips for tets in the flip queue.
@compute @workgroup_size(64)
fn flip_check(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let queue_size = params.x;

    if idx >= queue_size {
        return;
    }

    let tet_a = flip_queue[idx];

    // Skip dead tets
    let info_a = atomicLoad(&tet_info[tet_a]);
    if (info_a & TET_ALIVE) == 0u {
        return;
    }

    let tet_a_data = tets[tet_a];

    // Check each face of tet_a for Delaunay violation
    for (var face_a = 0u; face_a < 4u; face_a++) {
        let opp_packed = get_opp(tet_a, face_a);
        if opp_packed == INVALID {
            continue;
        }

        let tet_b = decode_opp_tet(opp_packed);
        let face_b = decode_opp_face(opp_packed);

        // Only process if tet_a < tet_b (avoid double processing)
        if tet_a >= tet_b {
            continue;
        }

        let info_b = atomicLoad(&tet_info[tet_b]);
        if (info_b & TET_ALIVE) == 0u {
            continue;
        }

        let tet_b_data = tets[tet_b];

        // Get the 5 involved vertices:
        // va = vertex of tet_a opposite the shared face
        // vb = vertex of tet_b opposite the shared face
        let va = tet_vertex(tet_a_data, face_a);
        let vb = tet_vertex(tet_b_data, face_b);

        // Get shared face vertices (the 3 vertices of tet_a not including va)
        var face_v: array<u32, 3>;
        var si = 0u;
        for (var i = 0u; i < 4u; i++) {
            let v = tet_vertex(tet_a_data, i);
            if v != va {
                face_v[si] = v;
                si++;
            }
        }

        let pa = points[va].xyz;
        let pb = points[vb].xyz;
        let p0 = points[face_v[0]].xyz;
        let p1 = points[face_v[1]].xyz;
        let p2 = points[face_v[2]].xyz;

        // Orient the tet (va, face_v[0], face_v[1], face_v[2]) positively
        let o = orient3d_simple(pa, p0, p1, p2);

        // Insphere test: is vb inside the circumsphere of tet_a?
        var is_inside: f32;
        if o > 0.0 {
            is_inside = insphere_simple(pa, p0, p1, p2, pb);
        } else {
            is_inside = -insphere_simple(pa, p0, p2, p1, pb);
        }

        if is_inside <= 0.0 {
            continue;
        }

        // === Detect 3-2 flip configuration ===
        // Check if 3 tets share the edge (va, vb) - CUDA: KerPredicates.cu:216-232
        let cor_vi = TET_VI_AS_SEEN_FROM[face_a];  // 3 other vertices of tet_a
        var is_flip32 = false;
        var bot_cor_ord_vi = 3u;  // Which edge is shared (0,1,2 for 3-2, 3 for 2-3)

        for (var i = 0u; i < 3u; i++) {
            let edge_vi = select(cor_vi.x, select(cor_vi.y, cor_vi.z, i == 2u), i >= 1u);
            let opp_edge = get_opp(tet_a, edge_vi);
            if opp_edge == INVALID {
                continue;
            }
            let edge_nei_tet = decode_opp_tet(opp_edge);
            let edge_nei_face = decode_opp_face(opp_edge);
            let edge_nei_data = tets[edge_nei_tet];
            let edge_nei_opp_v = tet_vertex(edge_nei_data, edge_nei_face);

            // Check if this neighbor also points to vb
            if edge_nei_opp_v == vb {
                // Found 3-2 configuration!
                is_flip32 = true;
                bot_cor_ord_vi = i;
                break;
            }
        }

        // Check flip validity: va-vb edge must intersect triangle(shared)
        let o0 = orient3d_simple(pa, pb, p0, p1);
        let o1 = orient3d_simple(pa, pb, p1, p2);
        let o2 = orient3d_simple(pa, pb, p2, p0);

        let s0 = sign(o0);
        let s1 = sign(o1);
        let s2 = sign(o2);

        if !((s0 >= 0 && s1 >= 0 && s2 >= 0) || (s0 <= 0 && s1 <= 0 && s2 <= 0)) {
            continue;
        }

        // === Try to lock both tets ===
        if !try_lock(tet_a) {
            continue;
        }
        if !try_lock(tet_b) {
            unlock(tet_a);
            continue;
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

            // Try to lock all 3 tets
            if !try_lock(tet_c) {
                unlock(tet_b);
                unlock(tet_a);
                continue;
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

            // Old face vertex indices for adjacency (CUDA lines 468-472)
            // oldFaceVi[ti][0..1] = the 2 face vertices from old tet ti
            let old_face_vi = array<vec2<u32>, 3>(
                vec2<u32>(bot_a_vi, bot_b_vi),    // Bot tet
                vec2<u32>(top_a_vi, top_b_vi),    // Top tet
                vec2<u32>(side_a_vi, side_b_vi),  // Side tet
            );

            // Create 2 new tets (CUDA lines 492-493)
            let bot_tet_v = va;  // Vertex opposite to shared face in bot tet
            let top_tet_v = vb;  // Vertex opposite to shared face in top tet
            var new_c0 = vec4<u32>(bot_cor, top_tet_v, bot_tet_v, bot_a);
            var new_c1 = vec4<u32>(bot_cor, bot_tet_v, top_tet_v, bot_b);

            // Orient positively
            let o_c0 = orient3d_simple(points[new_c0.x].xyz, points[new_c0.y].xyz, points[new_c0.z].xyz, points[new_c0.w].xyz);
            if o_c0 < 0.0 {
                let tmp = new_c0.z; new_c0.z = new_c0.w; new_c0.w = tmp;
            }
            let o_c1 = orient3d_simple(points[new_c1.x].xyz, points[new_c1.y].xyz, points[new_c1.z].xyz, points[new_c1.w].xyz);
            if o_c1 < 0.0 {
                let tmp = new_c1.z; new_c1.z = new_c1.w; new_c1.w = tmp;
            }

            // Write new tets (reuse tet_a and tet_b slots)
            tets[tet_a] = new_c0;
            tets[tet_b] = new_c1;

            // === External adjacency setup (CUDA lines 474-483) ===
            // For each old tet, set adjacency from old external faces to new tets
            let old_tets = array<u32, 3>(tet_a, tet_b, tet_c);
            let new_tets = array<u32, 3>(tet_a, tet_b, 0u);  // 0 unused for 3-2

            for (var ti = 0u; ti < 3u; ti++) {
                let old_tet = old_tets[ti];
                let old_tet_data = select(tet_a_data, select(tet_b_data, tet_c_data, ti == 2u), ti >= 1u);

                // Get old external neighbors for this tet's 2 faces
                let old_ext_0 = get_opp(old_tet, old_face_vi[ti].x);
                let old_ext_1 = get_opp(old_tet, old_face_vi[ti].y);

                // Determine which new tet gets which old face (CUDA line 477-478)
                // Flip32NewFaceVi[ti][0..1] tells us the new face indices
                let new_face_0 = FLIP32_NEW_FACE_VI[ti].x;
                let new_face_1 = FLIP32_NEW_FACE_VI[ti].y;

                // Which new tet? (CUDA line 477: newIdx[ti])
                let target_new_tet = select(tet_a, tet_b, ti == 1u);

                // Set new tet's external adjacency
                set_opp_at(target_new_tet, new_face_0, old_ext_0);
                set_opp_at(target_new_tet, new_face_1, old_ext_1);

                // Update back-pointers from neighbors
                if old_ext_0 != INVALID {
                    let nei_tet = decode_opp_tet(old_ext_0);
                    let nei_face = decode_opp_face(old_ext_0);
                    set_opp_at(nei_tet, nei_face, encode_opp(target_new_tet, new_face_0));
                }
                if old_ext_1 != INVALID {
                    let nei_tet = decode_opp_tet(old_ext_1);
                    let nei_face = decode_opp_face(old_ext_1);
                    set_opp_at(nei_tet, nei_face, encode_opp(target_new_tet, new_face_1));
                }
            }

            // === Internal adjacency between the 2 new tets ===
            // Find faces that connect new_c0 and new_c1
            // They share edge (bot_cor, bot_tet_v) or (bot_cor, top_tet_v)
            // new_c0 and new_c1 should be adjacent where they share the common face

            // Find which faces are opposite to the vertices not shared
            let c0_opposite_bot_a = find_local(new_c0, bot_a);
            let c1_opposite_bot_b = find_local(new_c1, bot_b);

            // These faces are adjacent to each other
            set_opp_at(tet_a, c0_opposite_bot_a, encode_opp(tet_b, c1_opposite_bot_b));
            set_opp_at(tet_b, c1_opposite_bot_b, encode_opp(tet_a, c0_opposite_bot_a));

            // Donate side tet back to free list (CUDA line 495-500)
            donate_tet(tet_c);

            // Mark new tets as alive
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
        // Two tets (tet_a, tet_b) sharing face (face_v[0..2]) →
        // Three tets sharing edge (va, vb).
        //
        // New tets:
        //   C0 = (va, vb, face_v[0], face_v[1])  at tet_a
        //   C1 = (va, vb, face_v[1], face_v[2])  at tet_b
        //   C2 = (va, vb, face_v[2], face_v[0])  at new_slot

        // Pop 1 free slot (2 tets → 3 tets = net +1)
        let new_slot = pop_free_slot();

        // Construct new tets and orient them positively
        var c0 = vec4<u32>(va, vb, face_v[0], face_v[1]);
        var c1 = vec4<u32>(va, vb, face_v[1], face_v[2]);
        var c2 = vec4<u32>(va, vb, face_v[2], face_v[0]);

        // Track orientation swaps (swapped[i] = true means we swapped [2],[3])
        var swapped = array<bool, 3>(false, false, false);

        let oc0 = orient3d_simple(points[c0.x].xyz, points[c0.y].xyz, points[c0.z].xyz, points[c0.w].xyz);
        if oc0 < 0.0 {
            let tmp = c0.z; c0.z = c0.w; c0.w = tmp;
            swapped[0] = true;
        }

        let oc1 = orient3d_simple(points[c1.x].xyz, points[c1.y].xyz, points[c1.z].xyz, points[c1.w].xyz);
        if oc1 < 0.0 {
            let tmp = c1.z; c1.z = c1.w; c1.w = tmp;
            swapped[1] = true;
        }

        let oc2 = orient3d_simple(points[c2.x].xyz, points[c2.y].xyz, points[c2.z].xyz, points[c2.w].xyz);
        if oc2 < 0.0 {
            let tmp = c2.z; c2.z = c2.w; c2.w = tmp;
            swapped[2] = true;
        }

        // Write new tets
        tets[tet_a] = c0;
        tets[tet_b] = c1;
        tets[new_slot] = c2;

        // === Internal adjacency ===
        // Before potential orientation swap, the shared faces between new tets are:
        //   C0↔C1: the face opposite to the shared vertex that C0 has but C1 doesn't,
        //           which is face_v[0] in C0 and face_v[2] in C1.
        //   C1↔C2: opposite face_v[1] in C1 and face_v[0] in C2.
        //   C2↔C0: opposite face_v[2] in C2 and face_v[1] in C0.
        //
        // After swap, we need to find the correct local indices.

        // Helper: find where a vertex ended up after potential swap.
        let c0_local_s0 = find_local(c0, face_v[0]);
        let c0_local_s1 = find_local(c0, face_v[1]);
        let c1_local_s1 = find_local(c1, face_v[1]);
        let c1_local_s2 = find_local(c1, face_v[2]);
        let c2_local_s2 = find_local(c2, face_v[2]);
        let c2_local_s0 = find_local(c2, face_v[0]);

        // === Internal adjacency ===
        // C0↔C1: C0 face opp face_v[0] <-> C1 face opp face_v[2]
        set_opp_at(tet_a, c0_local_s0, encode_opp(tet_b, c1_local_s2));
        set_opp_at(tet_b, c1_local_s2, encode_opp(tet_a, c0_local_s0));

        // C1↔C2: C1 face opp face_v[1] <-> C2 face opp face_v[0]
        set_opp_at(tet_b, c1_local_s1, encode_opp(new_slot, c2_local_s0));
        set_opp_at(new_slot, c2_local_s0, encode_opp(tet_b, c1_local_s1));

        // C2↔C0: C2 face opp face_v[2] <-> C0 face opp face_v[1]
        set_opp_at(new_slot, c2_local_s2, encode_opp(tet_a, c0_local_s1));
        set_opp_at(tet_a, c0_local_s1, encode_opp(new_slot, c2_local_s2));

        // === External adjacency ===
        let c0_local_va = find_local(c0, va);
        let c0_local_vb = find_local(c0, vb);
        let c1_local_va = find_local(c1, va);
        let c1_local_vb = find_local(c1, vb);
        let c2_local_va = find_local(c2, va);
        let c2_local_vb = find_local(c2, vb);

        // From tet_a: face opposite face_v[k] in tet_a -> new tet that doesn't have face_v[k]
        // s2 not in C0 -> C0 gets A's face-opp-s2, assigned to C0's face-opp-va
        let a_local_s2 = find_local(tet_a_data, face_v[2]);
        let ext_a_s2 = get_opp(tet_a, a_local_s2);
        set_opp_at(tet_a, c0_local_va, ext_a_s2);

        // s0 not in C1 -> C1 gets A's face-opp-s0
        let a_local_s0 = find_local(tet_a_data, face_v[0]);
        let ext_a_s0 = get_opp(tet_a, a_local_s0);
        set_opp_at(tet_b, c1_local_va, ext_a_s0);

        // s1 not in C2 -> C2 gets A's face-opp-s1
        let a_local_s1 = find_local(tet_a_data, face_v[1]);
        let ext_a_s1 = get_opp(tet_a, a_local_s1);
        set_opp_at(new_slot, c2_local_va, ext_a_s1);

        // From tet_b: face opposite face_v[k] in tet_b -> new tet that doesn't have face_v[k]
        let b_local_s2 = find_local(tet_b_data, face_v[2]);
        let ext_b_s2 = get_opp(tet_b, b_local_s2);
        set_opp_at(tet_a, c0_local_vb, ext_b_s2);

        let b_local_s0 = find_local(tet_b_data, face_v[0]);
        let ext_b_s0 = get_opp(tet_b, b_local_s0);
        set_opp_at(tet_b, c1_local_vb, ext_b_s0);

        let b_local_s1 = find_local(tet_b_data, face_v[1]);
        let ext_b_s1 = get_opp(tet_b, b_local_s1);
        set_opp_at(new_slot, c2_local_vb, ext_b_s1);

        // === Back-pointer fix ===
        // Update each external neighbor's tet_opp to point back to the correct new tet.
        if ext_a_s2 != INVALID {
            let n_tet = decode_opp_tet(ext_a_s2);
            let n_face = decode_opp_face(ext_a_s2);
            set_opp_at(n_tet, n_face, encode_opp(tet_a, c0_local_va));
        }
        if ext_a_s0 != INVALID {
            let n_tet = decode_opp_tet(ext_a_s0);
            let n_face = decode_opp_face(ext_a_s0);
            set_opp_at(n_tet, n_face, encode_opp(tet_b, c1_local_va));
        }
        if ext_a_s1 != INVALID {
            let n_tet = decode_opp_tet(ext_a_s1);
            let n_face = decode_opp_face(ext_a_s1);
            set_opp_at(n_tet, n_face, encode_opp(new_slot, c2_local_va));
        }
        if ext_b_s2 != INVALID {
            let n_tet = decode_opp_tet(ext_b_s2);
            let n_face = decode_opp_face(ext_b_s2);
            set_opp_at(n_tet, n_face, encode_opp(tet_a, c0_local_vb));
        }
        if ext_b_s0 != INVALID {
            let n_tet = decode_opp_tet(ext_b_s0);
            let n_face = decode_opp_face(ext_b_s0);
            set_opp_at(n_tet, n_face, encode_opp(tet_b, c1_local_vb));
        }
        if ext_b_s1 != INVALID {
            let n_tet = decode_opp_tet(ext_b_s1);
            let n_face = decode_opp_face(ext_b_s1);
            set_opp_at(n_tet, n_face, encode_opp(new_slot, c2_local_vb));
        }

        // Mark new tets as alive (clear lock)
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

        // One flip per thread — return after first success
        return;
    }
}
