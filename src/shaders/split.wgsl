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
// NOTE: Removed breadcrumbs and thread_debug to stay under 10 storage buffer limit
// @group(0) @binding(11) var<storage, read_write> breadcrumbs: array<u32>; // debug: progress tracking
// @group(0) @binding(12) var<storage, read_write> thread_debug: array<vec4<u32>>; // debug: 16 slots per thread

const INVALID: u32 = 0xFFFFFFFFu;
const TET_ALIVE: u32 = 1u;
const TET_CHANGED: u32 = 2u;
const COUNTER_FREE: u32 = 0u;
const COUNTER_ACTIVE: u32 = 1u;
const MEAN_VERTEX_DEGREE: u32 = 8u;

// Breadcrumb constants for tracking execution (DISABLED)
const CRUMB_START: u32 = 1u;
const CRUMB_READ_INSERT: u32 = 2u;
const CRUMB_AFTER_ALLOC: u32 = 3u;
const CRUMB_BEFORE_WRITE: u32 = 4u;
const CRUMB_AFTER_WRITE_T0: u32 = 5u;
const CRUMB_AFTER_WRITE_ALL: u32 = 6u;
const CRUMB_AFTER_MARK_DEAD: u32 = 7u;
const CRUMB_AFTER_ADJACENCY: u32 = 8u;
const CRUMB_COMPLETE: u32 = 99u;

// Debug helper functions (DISABLED - no-ops to stay under buffer limit)
fn breadcrumb(tid: u32, crumb: u32) {
    // breadcrumbs[tid] = crumb;
}

fn debug_slot(tid: u32, slot: u32, values: vec4<u32>) {
    // thread_debug[tid * 16u + slot] = values;
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

    // Write the 4 new tets using CUDA's TetViAsSeenFrom permutation
    // CUDA: newTet[vi] = { tet._v[TetViAsSeenFrom[vi][0]],
    //                       tet._v[TetViAsSeenFrom[vi][1]],
    //                       tet._v[TetViAsSeenFrom[vi][2]],
    //                       splitVertex }
    // TetViAsSeenFrom: {1,3,2}, {0,2,3}, {0,3,1}, {0,1,2}
    tets[t0] = vec4<u32>(v1, v3, v2, p);  // vi=0: AsSeenFrom[0] = {1,3,2}
    breadcrumb(tid, CRUMB_AFTER_WRITE_T0);

    tets[t1] = vec4<u32>(v0, v2, v3, p);  // vi=1: AsSeenFrom[1] = {0,2,3}
    tets[t2] = vec4<u32>(v0, v3, v1, p);  // vi=2: AsSeenFrom[2] = {0,3,1}
    tets[t3] = vec4<u32>(v0, v1, v2, p);  // vi=3: AsSeenFrom[3] = {0,1,2}

    breadcrumb(tid, CRUMB_AFTER_WRITE_ALL);

    // Mark old tet as dead
    tet_info[old_tet] = 0u;

    breadcrumb(tid, CRUMB_AFTER_MARK_DEAD);

    // Internal adjacency using CUDA's IntSplitFaceOpp table:
    // IntSplitFaceOpp[4][6] = {
    //   {1, 0, 3, 0, 2, 0},  // vi=0
    //   {0, 0, 2, 2, 3, 1},  // vi=1
    //   {0, 2, 3, 2, 1, 1},  // vi=2
    //   {0, 1, 1, 2, 2, 1}   // vi=3
    // }
    // For each tet vi:
    //   face 0 -> newTet[IntSplitFaceOpp[vi][0]], face IntSplitFaceOpp[vi][1]
    //   face 1 -> newTet[IntSplitFaceOpp[vi][2]], face IntSplitFaceOpp[vi][3]
    //   face 2 -> newTet[IntSplitFaceOpp[vi][4]], face IntSplitFaceOpp[vi][5]
    //   face 3 -> external (original opposite face)

    // T0 (vi=0) adjacency: IntSplitFaceOpp[0] = {1, 0, 3, 0, 2, 0}
    set_opp_at(t0, 0u, encode_opp(t1, 0u));   // face 0: T1's face 0
    set_opp_at(t0, 1u, encode_opp(t3, 0u));   // face 1: T3's face 0
    set_opp_at(t0, 2u, encode_opp(t2, 0u));   // face 2: T2's face 0
    set_opp_at(t0, 3u, orig_opp_0);           // face 3: external (was opp v0)

    // T1 (vi=1) adjacency: IntSplitFaceOpp[1] = {0, 0, 2, 2, 3, 1}
    set_opp_at(t1, 0u, encode_opp(t0, 0u));   // face 0: T0's face 0
    set_opp_at(t1, 1u, encode_opp(t2, 2u));   // face 1: T2's face 2
    set_opp_at(t1, 2u, encode_opp(t3, 1u));   // face 2: T3's face 1
    set_opp_at(t1, 3u, orig_opp_1);           // face 3: external (was opp v1)

    // T2 (vi=2) adjacency: IntSplitFaceOpp[2] = {0, 2, 3, 2, 1, 1}
    set_opp_at(t2, 0u, encode_opp(t0, 2u));   // face 0: T0's face 2
    set_opp_at(t2, 1u, encode_opp(t3, 2u));   // face 1: T3's face 2
    set_opp_at(t2, 2u, encode_opp(t1, 1u));   // face 2: T1's face 1
    set_opp_at(t2, 3u, orig_opp_2);           // face 3: external (was opp v2)

    // T3 (vi=3) adjacency: IntSplitFaceOpp[3] = {0, 1, 1, 2, 2, 1}
    set_opp_at(t3, 0u, encode_opp(t0, 1u));   // face 0: T0's face 1
    set_opp_at(t3, 1u, encode_opp(t1, 2u));   // face 1: T1's face 2
    set_opp_at(t3, 2u, encode_opp(t2, 1u));   // face 2: T2's face 1
    set_opp_at(t3, 3u, orig_opp_3);           // face 3: external (was opp v3)

    // Update external neighbours to point back to us.
    // Uses tet_to_vert to detect concurrent splits (CUDA: KerDivision.cu:140-162)
    let ext_opps = array<u32, 4>(orig_opp_0, orig_opp_1, orig_opp_2, orig_opp_3);
    let new_tets = array<u32, 4>(t0, t1, t2, t3);

    for (var k = 0u; k < 4u; k++) {
        let ext_opp = ext_opps[k];

        if ext_opp != INVALID {
            var nei_tet = decode_opp_tet(ext_opp);
            var nei_face = decode_opp_face(ext_opp);

            // Check if neighbor is splitting concurrently
            let nei_split_idx = tet_to_vert[nei_tet];

            if nei_split_idx != INVALID {
                // Neighbor has split - use free_arr to find correct new tet
                let nei_split_vert = insert_list[nei_split_idx].y;  // vertex being inserted
                let nei_free_idx = (nei_split_vert + 1u) * MEAN_VERTEX_DEGREE - 1u;

                nei_tet = free_arr[nei_free_idx - nei_face];
                nei_face = 3u;  // External faces become face 3 after split
            }

            // Point neighbor back to this new tet (safe whether split or not)
            set_opp_at(nei_tet, nei_face, encode_opp(new_tets[k], k));
        }
    }

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
