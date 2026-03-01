// Kernel: Split a tetrahedron into 4 new tetrahedra by inserting a point.
//
// Given tet T with vertices (v0, v1, v2, v3) and new point P:
//   T0 = (P, v1, v2, v3)  — replaces original tet
//   T1 = (v0, P, v2, v3)  — new
//   T2 = (v0, v1, P, v3)  — new
//   T3 = (v0, v1, v2, P)  — new
//
// Each new tet Tk replaces vertex k with P. This preserves orientation if
// the original tet was positively oriented.
//
// Adjacency update:
//   - Internal faces between T0..T3 link to each other
//   - External faces keep the original neighbour (but the neighbour needs updating too)
//
// tet_opp is a flat array<atomic<u32>> indexed as [tet_idx * 4 + face].
//
// Dispatch: ceil(num_insertions / 64)

@group(0) @binding(0) var<storage, read_write> tets: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> tet_opp: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> tet_info: array<u32>;
@group(0) @binding(3) var<storage, read_write> vert_tet: array<u32>;
@group(0) @binding(4) var<storage, read> insert_list: array<vec2<u32>>; // (tet_idx, vert_idx)
@group(0) @binding(5) var<storage, read_write> free_arr: array<u32>;
@group(0) @binding(6) var<storage, read_write> vert_free_arr: array<u32>;
@group(0) @binding(7) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> flip_queue: array<u32>; // tets needing flip check
@group(0) @binding(9) var<storage, read_write> tet_to_vert: array<u32>; // maps old_tet_idx -> vertex being inserted (or INVALID)
@group(0) @binding(10) var<uniform> params: vec4<u32>; // x = num_insertions
@group(0) @binding(11) var<storage, read_write> breadcrumbs: array<u32>; // debug: progress tracking
@group(0) @binding(12) var<storage, read_write> thread_debug: array<vec4<u32>>; // debug: 16 slots per thread

const INVALID: u32 = 0xFFFFFFFFu;
const TET_ALIVE: u32 = 1u;
const TET_CHANGED: u32 = 2u;
const COUNTER_FREE: u32 = 0u;
const COUNTER_ACTIVE: u32 = 1u;
const MEAN_VERTEX_DEGREE: u32 = 64u;

// Breadcrumb constants for tracking execution
const CRUMB_START: u32 = 1u;
const CRUMB_READ_INSERT: u32 = 2u;
const CRUMB_AFTER_ALLOC: u32 = 3u;
const CRUMB_BEFORE_WRITE: u32 = 4u;
const CRUMB_AFTER_WRITE_T0: u32 = 5u;
const CRUMB_AFTER_WRITE_ALL: u32 = 6u;
const CRUMB_AFTER_MARK_DEAD: u32 = 7u;
const CRUMB_AFTER_ADJACENCY: u32 = 8u;
const CRUMB_COMPLETE: u32 = 99u;

// Debug helper functions
fn breadcrumb(tid: u32, crumb: u32) {
    breadcrumbs[tid] = crumb;
}

fn debug_slot(tid: u32, slot: u32, values: vec4<u32>) {
    thread_debug[tid * 16u + slot] = values;
}

fn encode_opp(tet_idx: u32, face: u32) -> u32 {
    return (tet_idx << 2u) | (face & 3u);
}

fn decode_opp_tet(packed: u32) -> u32 {
    return packed >> 2u;
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

// Get 4 free tet slots
fn get_free_slots_4tet(vertex: u32) -> vec4<u32> {
    let count = vert_free_arr[vertex];

    // Check if enough free slots
    if count < 4u {
        return vec4<u32>(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
    }

    // Compute top of available slots based on current count
    let block_base = vertex * MEAN_VERTEX_DEGREE;
    let top = block_base + count - 1u;

    var slots: vec4<u32>;
    slots.x = free_arr[top];
    slots.y = free_arr[top - 1u];
    slots.z = free_arr[top - 2u];
    slots.w = free_arr[top - 3u];

    vert_free_arr[vertex] = count - 4u;
    return slots;
}

@compute @workgroup_size(64)
fn split_tetra(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let tid = gid.x;  // Thread ID for debugging
    let idx = tid;
    let num_insertions = params.x;

    if idx >= num_insertions {
        return;
    }

    breadcrumb(tid, CRUMB_START);

    let insert = insert_list[idx];
    let old_tet = insert.x;
    let p = insert.y;

    breadcrumb(tid, CRUMB_READ_INSERT);
    debug_slot(tid, 0u, vec4<u32>(old_tet, p, 0u, 0u));

    // Read original tet vertices and adjacency
    let orig = tets[old_tet];
    let orig_opp_0 = get_opp(old_tet, 0u);
    let orig_opp_1 = get_opp(old_tet, 1u);
    let orig_opp_2 = get_opp(old_tet, 2u);
    let orig_opp_3 = get_opp(old_tet, 3u);
    let v0 = orig.x;
    let v1 = orig.y;
    let v2 = orig.z;
    let v3 = orig.w;

    // Allocate 4 new tet slots
    let new_slots = get_free_slots_4tet(p);
    let t0 = new_slots.x;
    let t1 = new_slots.y;
    let t2 = new_slots.z;
    let t3 = new_slots.w;

    breadcrumb(tid, CRUMB_AFTER_ALLOC);
    debug_slot(tid, 1u, new_slots);

    // Check allocation succeeded
    if t0 == INVALID {
        breadcrumb(tid, 0xDEADu);  // Allocation failed marker
        return;
    }

    breadcrumb(tid, CRUMB_BEFORE_WRITE);

    // Write the 4 new tets
    tets[t0] = vec4<u32>(p, v1, v2, v3);
    breadcrumb(tid, CRUMB_AFTER_WRITE_T0);

    tets[t1] = vec4<u32>(v0, p, v2, v3);
    tets[t2] = vec4<u32>(v0, v1, p, v3);
    tets[t3] = vec4<u32>(v0, v1, v2, p);

    breadcrumb(tid, CRUMB_AFTER_WRITE_ALL);

    // Mark old tet as dead
    tet_info[old_tet] = 0u;

    breadcrumb(tid, CRUMB_AFTER_MARK_DEAD);

    // Internal adjacency:
    // Tk's face opposite P (face k in Tk) connects to external neighbour.
    // Tk's face opposite vi (i != k) connects to Ti.
    //
    // For T0 (P, v1, v2, v3):
    //   face 0 (opp P) = external = orig_opp[0] (was opp v0)
    //   face 1 (opp v1) = T1's face 0 (opp v0 in T1... wait)
    //
    // Let's think about this more carefully.
    // In Tk, vertex k has been replaced by P.
    // The face opposite to P in Tk is face k (the face that doesn't include P).
    //   This face has the same 3 vertices as face k in the original tet.
    //   So its external neighbour is orig_opp[k].
    //
    // The face opposite to vi (i != k) in Tk includes P and the other 2 vertices.
    //   This face is shared with Ti (where vi was replaced by P).
    //   Specifically, face i in Tk connects to face k in Ti.

    // T0 adjacency:
    set_opp_at(t0, 0u, orig_opp_0);           // face 0: external (was opp v0)
    set_opp_at(t0, 1u, encode_opp(t1, 0u));   // face 1: T1's face 0
    set_opp_at(t0, 2u, encode_opp(t2, 0u));   // face 2: T2's face 0
    set_opp_at(t0, 3u, encode_opp(t3, 0u));   // face 3: T3's face 0

    // T1 adjacency:
    set_opp_at(t1, 0u, encode_opp(t0, 1u));   // face 0: T0's face 1
    set_opp_at(t1, 1u, orig_opp_1);           // face 1: external (was opp v1)
    set_opp_at(t1, 2u, encode_opp(t2, 1u));   // face 2: T2's face 1
    set_opp_at(t1, 3u, encode_opp(t3, 1u));   // face 3: T3's face 1

    // T2 adjacency:
    set_opp_at(t2, 0u, encode_opp(t0, 2u));   // face 0: T0's face 2
    set_opp_at(t2, 1u, encode_opp(t1, 2u));   // face 1: T1's face 2
    set_opp_at(t2, 2u, orig_opp_2);           // face 2: external (was opp v2)
    set_opp_at(t2, 3u, encode_opp(t3, 2u));   // face 3: T3's face 2

    // T3 adjacency:
    set_opp_at(t3, 0u, encode_opp(t0, 3u));   // face 0: T0's face 3
    set_opp_at(t3, 1u, encode_opp(t1, 3u));   // face 1: T1's face 3
    set_opp_at(t3, 2u, encode_opp(t2, 3u));   // face 2: T2's face 3
    set_opp_at(t3, 3u, orig_opp_3);           // face 3: external (was opp v3)

    // Update external neighbours to point back to us.
    // TEMPORARILY DISABLED: This causes race conditions when multiple threads split the same tet
    // TODO: Fix voting system to ensure only 1 winner per tet, OR implement atomic claiming
    /*
    let ext_opps = array<u32, 4>(orig_opp_0, orig_opp_1, orig_opp_2, orig_opp_3);
    let new_tets = array<u32, 4>(t0, t1, t2, t3);

    for (var k = 0u; k < 4u; k++) {
        let ext_opp = ext_opps[k];

        if ext_opp != INVALID {
            var nei_tet = decode_opp_tet(ext_opp);
            var nei_face = decode_opp_face(ext_opp);

            set_opp_at(nei_tet, nei_face, encode_opp(new_tets[k], k));
        }
    }
    */

    breadcrumb(tid, CRUMB_AFTER_ADJACENCY);

    // Mark all 4 tets as alive + changed (need flip checking)
    tet_info[t0] = TET_ALIVE | TET_CHANGED;
    tet_info[t1] = TET_ALIVE | TET_CHANGED;
    tet_info[t2] = TET_ALIVE | TET_CHANGED;
    tet_info[t3] = TET_ALIVE | TET_CHANGED;

    // Clear tet_to_vert for the newly allocated tets to prevent stale values
    tet_to_vert[t0] = INVALID;
    tet_to_vert[t1] = INVALID;
    tet_to_vert[t2] = INVALID;
    tet_to_vert[t3] = INVALID;

    // Update active count (+4 new, -1 old = net +3)
    atomicAdd(&counters[COUNTER_ACTIVE], 3u);

    // Map the inserted vertex to one of its tets
    vert_tet[p] = t0;

    // Add all 4 tets to flip queue (they need Delaunay checking)
    let fq_base = idx * 4u;
    flip_queue[fq_base] = t0;
    flip_queue[fq_base + 1u] = t1;
    flip_queue[fq_base + 2u] = t2;
    flip_queue[fq_base + 3u] = t3;

    breadcrumb(tid, CRUMB_COMPLETE);
    debug_slot(tid, 2u, vec4<u32>(t0, t1, t2, t3));  // Final allocated tets
}
