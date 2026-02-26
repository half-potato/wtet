// Kernels: Delaunay flipping (2-3 flips) with full execution.
//
// After point insertion, check each new tet's external faces for Delaunay
// violation (insphere test). If violated, perform a bistellar 2-3 flip.
//
// 2-3 flip: Two tets sharing a face → three tets sharing an edge.
//   T_abc_d + T_abc_e → T_ab_de + T_bc_de + T_ca_de
//
// Uses bilateral CAS locking to prevent concurrent modification of the
// same tets by different threads.

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> tets: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> tet_opp: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> tet_info: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> free_stack: array<u32>;
@group(0) @binding(5) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> flip_queue: array<u32>;
@group(0) @binding(7) var<storage, read_write> flip_queue_next: array<u32>;
@group(0) @binding(8) var<storage, read_write> flip_count: array<atomic<u32>>; // [0] = next queue size
@group(0) @binding(9) var<uniform> params: vec4<u32>; // x = queue_size

const INVALID: u32 = 0xFFFFFFFFu;
const TET_ALIVE: u32 = 1u;
const TET_CHANGED: u32 = 2u;
const TET_CHECKED: u32 = 4u;
const TET_LOCKED: u32 = 8u;
const COUNTER_FREE: u32 = 0u;
const COUNTER_ACTIVE: u32 = 1u;

fn encode_opp(tet_idx: u32, face: u32) -> u32 {
    return (tet_idx << 2u) | (face & 3u);
}

fn decode_opp_tet(packed: u32) -> u32 {
    return packed >> 2u;
}

fn decode_opp_face(packed: u32) -> u32 {
    return packed & 3u;
}

fn tet_vertex(tet: vec4<u32>, i: u32) -> u32 {
    switch i {
        case 0u: { return tet.x; }
        case 1u: { return tet.y; }
        case 2u: { return tet.z; }
        default: { return tet.w; }
    }
}

fn opp_entry(opp: vec4<u32>, i: u32) -> u32 {
    switch i {
        case 0u: { return opp.x; }
        case 1u: { return opp.y; }
        case 2u: { return opp.z; }
        default: { return opp.w; }
    }
}

fn set_opp_entry(opp: vec4<u32>, i: u32, val: u32) -> vec4<u32> {
    var r = opp;
    switch i {
        case 0u: { r.x = val; }
        case 1u: { r.y = val; }
        case 2u: { r.z = val; }
        default: { r.w = val; }
    }
    return r;
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

// Pop 1 free slot from the free stack.
fn pop_free_slot() -> u32 {
    let old_free = atomicSub(&counters[COUNTER_FREE], 1u);
    return free_stack[old_free - 1u];
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
    let tet_a_opp = tet_opp[tet_a];

    // Check each face of tet_a for Delaunay violation
    for (var face_a = 0u; face_a < 4u; face_a++) {
        let opp_packed = opp_entry(tet_a_opp, face_a);
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

        var opp_c0 = vec4<u32>(INVALID, INVALID, INVALID, INVALID);
        var opp_c1 = vec4<u32>(INVALID, INVALID, INVALID, INVALID);
        var opp_c2 = vec4<u32>(INVALID, INVALID, INVALID, INVALID);

        // C0↔C1: C0 face opp face_v[0] <-> C1 face opp face_v[2]
        // (The face opposite face_v[0] in C0 = face at local index of face_v[0] in C0)
        opp_c0 = set_opp_entry(opp_c0, c0_local_s0, encode_opp(tet_b, c1_local_s2));
        opp_c1 = set_opp_entry(opp_c1, c1_local_s2, encode_opp(tet_a, c0_local_s0));

        // C1↔C2: C1 face opp face_v[1] <-> C2 face opp face_v[0]
        opp_c1 = set_opp_entry(opp_c1, c1_local_s1, encode_opp(new_slot, c2_local_s0));
        opp_c2 = set_opp_entry(opp_c2, c2_local_s0, encode_opp(tet_b, c1_local_s1));

        // C2↔C0: C2 face opp face_v[2] <-> C0 face opp face_v[1]
        opp_c2 = set_opp_entry(opp_c2, c2_local_s2, encode_opp(tet_a, c0_local_s1));
        opp_c0 = set_opp_entry(opp_c0, c0_local_s1, encode_opp(new_slot, c2_local_s2));

        // === External adjacency ===
        // Each new tet has 2 external faces: one from tet_a and one from tet_b.
        // Tet_a had 4 faces. Face face_a was the shared face (now gone).
        // The other 3 faces of tet_a each excluded one of face_v[0,1,2].
        // Similarly for tet_b.
        //
        // For each shared vertex sk:
        //   - In tet_a, the face opposite sk has local index find_local(tet_a_data, sk).
        //     This face maps to the new tet that does NOT contain sk:
        //       If sk == face_v[0]: goes to C1 (which has face_v[1],face_v[2] but not face_v[0])
        //         Wait - C0 has face_v[0],face_v[1]. C1 has face_v[1],face_v[2]. C2 has face_v[2],face_v[0].
        //       Actually: C0 has {va,vb,s0,s1}, C1 has {va,vb,s1,s2}, C2 has {va,vb,s2,s0}.
        //       Face NOT containing sk (which came from tet_a side):
        //         s0 is NOT in C1 → A's face-opp-s0 → C1's face-opp-va
        //         s1 is NOT in C2 → A's face-opp-s1 → C2's face-opp-va
        //         s2 is NOT in C0 → A's face-opp-s2 → C0's face-opp-va
        //
        //       From tet_b side:
        //         s0 is NOT in C1 → B's face-opp-s0 → C1's face-opp-vb
        //         s1 is NOT in C2 → B's face-opp-s1 → C2's face-opp-vb
        //         s2 is NOT in C0 → B's face-opp-s2 → C0's face-opp-vb

        // Re-read adjacency (may have been partially overwritten in previous iterations but
        // we read these at the top of the function)
        let tet_a_opp_local = tet_a_opp;
        let tet_b_opp = tet_opp[tet_b];

        let c0_local_va = find_local(c0, va);
        let c0_local_vb = find_local(c0, vb);
        let c1_local_va = find_local(c1, va);
        let c1_local_vb = find_local(c1, vb);
        let c2_local_va = find_local(c2, va);
        let c2_local_vb = find_local(c2, vb);

        // From tet_a: face opposite face_v[k] in tet_a -> new tet that doesn't have face_v[k]
        // s2 not in C0 -> C0 gets A's face-opp-s2, assigned to C0's face-opp-va
        let a_local_s2 = find_local(tet_a_data, face_v[2]);
        let ext_a_s2 = opp_entry(tet_a_opp_local, a_local_s2);
        opp_c0 = set_opp_entry(opp_c0, c0_local_va, ext_a_s2);

        // s0 not in C1 -> C1 gets A's face-opp-s0
        let a_local_s0 = find_local(tet_a_data, face_v[0]);
        let ext_a_s0 = opp_entry(tet_a_opp_local, a_local_s0);
        opp_c1 = set_opp_entry(opp_c1, c1_local_va, ext_a_s0);

        // s1 not in C2 -> C2 gets A's face-opp-s1
        let a_local_s1 = find_local(tet_a_data, face_v[1]);
        let ext_a_s1 = opp_entry(tet_a_opp_local, a_local_s1);
        opp_c2 = set_opp_entry(opp_c2, c2_local_va, ext_a_s1);

        // From tet_b: face opposite face_v[k] in tet_b -> new tet that doesn't have face_v[k]
        let b_local_s2 = find_local(tet_b_data, face_v[2]);
        let ext_b_s2 = opp_entry(tet_b_opp, b_local_s2);
        opp_c0 = set_opp_entry(opp_c0, c0_local_vb, ext_b_s2);

        let b_local_s0 = find_local(tet_b_data, face_v[0]);
        let ext_b_s0 = opp_entry(tet_b_opp, b_local_s0);
        opp_c1 = set_opp_entry(opp_c1, c1_local_vb, ext_b_s0);

        let b_local_s1 = find_local(tet_b_data, face_v[1]);
        let ext_b_s1 = opp_entry(tet_b_opp, b_local_s1);
        opp_c2 = set_opp_entry(opp_c2, c2_local_vb, ext_b_s1);

        // Write adjacency
        tet_opp[tet_a] = opp_c0;
        tet_opp[tet_b] = opp_c1;
        tet_opp[new_slot] = opp_c2;

        // === Back-pointer fix ===
        // Update each external neighbor's tet_opp to point back to the correct new tet.
        if ext_a_s2 != INVALID {
            let n_tet = decode_opp_tet(ext_a_s2);
            let n_face = decode_opp_face(ext_a_s2);
            var n_opp = tet_opp[n_tet];
            n_opp = set_opp_entry(n_opp, n_face, encode_opp(tet_a, c0_local_va));
            tet_opp[n_tet] = n_opp;
        }
        if ext_a_s0 != INVALID {
            let n_tet = decode_opp_tet(ext_a_s0);
            let n_face = decode_opp_face(ext_a_s0);
            var n_opp = tet_opp[n_tet];
            n_opp = set_opp_entry(n_opp, n_face, encode_opp(tet_b, c1_local_va));
            tet_opp[n_tet] = n_opp;
        }
        if ext_a_s1 != INVALID {
            let n_tet = decode_opp_tet(ext_a_s1);
            let n_face = decode_opp_face(ext_a_s1);
            var n_opp = tet_opp[n_tet];
            n_opp = set_opp_entry(n_opp, n_face, encode_opp(new_slot, c2_local_va));
            tet_opp[n_tet] = n_opp;
        }
        if ext_b_s2 != INVALID {
            let n_tet = decode_opp_tet(ext_b_s2);
            let n_face = decode_opp_face(ext_b_s2);
            var n_opp = tet_opp[n_tet];
            n_opp = set_opp_entry(n_opp, n_face, encode_opp(tet_a, c0_local_vb));
            tet_opp[n_tet] = n_opp;
        }
        if ext_b_s0 != INVALID {
            let n_tet = decode_opp_tet(ext_b_s0);
            let n_face = decode_opp_face(ext_b_s0);
            var n_opp = tet_opp[n_tet];
            n_opp = set_opp_entry(n_opp, n_face, encode_opp(tet_b, c1_local_vb));
            tet_opp[n_tet] = n_opp;
        }
        if ext_b_s1 != INVALID {
            let n_tet = decode_opp_tet(ext_b_s1);
            let n_face = decode_opp_face(ext_b_s1);
            var n_opp = tet_opp[n_tet];
            n_opp = set_opp_entry(n_opp, n_face, encode_opp(new_slot, c2_local_vb));
            tet_opp[n_tet] = n_opp;
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
