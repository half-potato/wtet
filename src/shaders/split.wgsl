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
@group(0) @binding(6) var<storage, read_write> vert_free_arr: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> flip_queue: array<u32>; // tets needing flip check
@group(0) @binding(9) var<storage, read_write> tet_to_vert: array<u32>; // maps old_tet_idx -> vertex being inserted (or INVALID)
@group(0) @binding(10) var<uniform> params: vec4<u32>; // x = num_insertions, y = inf_idx, z = current_tet_num
@group(0) @binding(11) var<storage, read> block_owner: array<u32>; // Pre-computed block ownership (Issue #3 fix)
@group(0) @binding(12) var<storage, read> uninserted: array<u32>; // Maps position -> vertex ID
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
// CRITICAL: Must match CUDA's pre-allocated approach (KerDivision.cu:120-124)
// CUDA allocates 4 CONSECUTIVE pre-reserved slots from the END of the vertex's block
fn get_free_slots_4tet(vertex: u32) -> vec4<u32> {
    // CUDA line 120: const int newIdx = ( splitVertex + 1 ) * MeanVertDegree - 1;
    let base_idx = (vertex + 1u) * MEAN_VERTEX_DEGREE - 1u;

    // CUDA line 121: const int newTetIdx[4] = { freeArr[newIdx], freeArr[newIdx-1], freeArr[newIdx-2], freeArr[newIdx-3] };
    let slots = vec4<u32>(
        free_arr[base_idx],
        free_arr[base_idx - 1u],
        free_arr[base_idx - 2u],
        free_arr[base_idx - 3u]
    );

    // CUDA line 124: vertFreeArr[ splitVertex ] -= 4;
    atomicSub(&vert_free_arr[vertex], 4u);

    return slots;
}

@compute @workgroup_size(64)
fn split_tetra(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let tid = gid.x;  // Thread ID for debugging
    let idx = tid;

    // DIAGNOSTIC: Write a sentinel value to prove shader executed
    if idx == 0u {
        tet_info[5] = 999u;
    }

    let num_insertions = params.x;
    let inf_idx = params.y;

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

    // Get the actual vertex ID (p is position in uninserted array, not vertex ID!)
    let vertex = uninserted[p];

    // Allocate 4 new tet slots from vertex's pre-reserved block
    // Uses CUDA's simple approach: read 4 consecutive slots from end of block
    let new_slots = get_free_slots_4tet(vertex);
    let t0 = new_slots.x;
    let t1 = new_slots.y;
    let t2 = new_slots.z;
    let t3 = new_slots.w;

    // DIAGNOSTIC: Write allocated tet indices to tet_info[6-9]
    if idx == 0u {
        if t0 == 0u {
            tet_info[6] = 0xAAAAu;  // t0 is 0
        } else {
            tet_info[6] = t0;
        }
        tet_info[7] = t1;
        tet_info[9] = t3;
    }

    breadcrumb(tid, CRUMB_AFTER_ALLOC);
    debug_slot(tid, 1u, new_slots);

    // Check allocation succeeded
    if t0 == INVALID {
        // DIAGNOSTIC: Mark allocation failure
        if idx == 0u {
            tet_info[7] = 0xDEADu;
        }
        breadcrumb(tid, 0xDEADu);  // Allocation failed marker
        return;
    }

    // DIAGNOSTIC: Mark allocation success
    if idx == 0u {
        tet_info[8] = 0xBEEFu;
    }

    breadcrumb(tid, CRUMB_BEFORE_WRITE);

    // Write the 4 new tets using CUDA's TetViAsSeenFrom permutation
    // CUDA: newTet[vi] = { tet._v[TetViAsSeenFrom[vi][0]],
    //                       tet._v[TetViAsSeenFrom[vi][1]],
    //                       tet._v[TetViAsSeenFrom[vi][2]],
    //                       splitVertex }
    // TetViAsSeenFrom: {1,3,2}, {0,2,3}, {0,3,1}, {0,1,2}
    tets[t0] = vec4<u32>(v1, v3, v2, vertex);  // vi=0: AsSeenFrom[0] = {1,3,2}
    breadcrumb(tid, CRUMB_AFTER_WRITE_T0);

    tets[t1] = vec4<u32>(v0, v2, v3, vertex);  // vi=1: AsSeenFrom[1] = {0,2,3}
    tets[t2] = vec4<u32>(v0, v3, v1, vertex);  // vi=2: AsSeenFrom[2] = {0,3,1}
    tets[t3] = vec4<u32>(v0, v1, v2, vertex);  // vi=3: AsSeenFrom[3] = {0,1,2}

    breadcrumb(tid, CRUMB_AFTER_WRITE_ALL);

    // ═══════════════════════════════════════════════════════════════════════════
    // Donate old tet back to its owner's free list
    // ═══════════════════════════════════════════════════════════════════════════
    // DONATION LOGIC - Port of CUDA KerDivision.cu:181-188
    //
    // CUDA:
    //   const int blkIdx  = tetIdx / MeanVertDegree;
    //   const int vertIdx = ( blkIdx < insVertVec._num ) ? insVertVec._arr[ blkIdx ] : infIdx;
    //   const int freeIdx = atomicAdd( &vertFreeArr[ vertIdx ], 1 );
    //   freeArr[ vertIdx * MeanVertDegree + freeIdx ] = tetIdx;
    //   setTetAliveState( tetInfoArr[ tetIdx ], false );
    //
    // Issue #3 Fix: Use pre-computed block_owner lookup (matches CUDA's array lookup)
    //
    // Calculate which block this tet belongs to
    let blk_idx = old_tet / MEAN_VERTEX_DEGREE;

    // Lookup owner from pre-computed buffer (replaces CUDA's insVertVec._arr[blkIdx])
    let owner_vertex = block_owner[blk_idx];

    // Atomically increment this vertex's free count and get the slot index
    let free_slot_idx = atomicAdd(&vert_free_arr[owner_vertex], 1u);

    // Store old_tet in the owner's free list
    free_arr[owner_vertex * MEAN_VERTEX_DEGREE + free_slot_idx] = old_tet;

    // Mark old tet as dead
    tet_info[old_tet] = 0u;
    // ═══════════════════════════════════════════════════════════════════════════

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
    // face 3: external - set below after detecting concurrent splits

    // T1 (vi=1) adjacency: IntSplitFaceOpp[1] = {0, 0, 2, 2, 3, 1}
    set_opp_at(t1, 0u, encode_opp(t0, 0u));   // face 0: T0's face 0
    set_opp_at(t1, 1u, encode_opp(t2, 2u));   // face 1: T2's face 2
    set_opp_at(t1, 2u, encode_opp(t3, 1u));   // face 2: T3's face 1
    // face 3: external - set below after detecting concurrent splits

    // T2 (vi=2) adjacency: IntSplitFaceOpp[2] = {0, 2, 3, 2, 1, 1}
    set_opp_at(t2, 0u, encode_opp(t0, 2u));   // face 0: T0's face 2
    set_opp_at(t2, 1u, encode_opp(t3, 2u));   // face 1: T3's face 2
    set_opp_at(t2, 2u, encode_opp(t1, 1u));   // face 2: T1's face 1
    // face 3: external - set below after detecting concurrent splits

    // T3 (vi=3) adjacency: IntSplitFaceOpp[3] = {0, 1, 1, 2, 2, 1}
    set_opp_at(t3, 0u, encode_opp(t0, 1u));   // face 0: T0's face 1
    set_opp_at(t3, 1u, encode_opp(t1, 2u));   // face 1: T1's face 2
    set_opp_at(t3, 2u, encode_opp(t2, 1u));   // face 2: T2's face 1
    // face 3: external - set below after detecting concurrent splits

    // ═══════════════════════════════════════════════════════════════════════════
    // Set external adjacency (face 3) for each of the 4 new tets
    // ═══════════════════════════════════════════════════════════════════════════
    // Port of CUDA KerDivision.cu:139-163
    //
    // CRITICAL: Must detect concurrent splits BEFORE setting face 3!
    // If a neighbor has split concurrently, face 3 must point to the correct
    // new tet (not the old dead tet).
    //
    // CUDA:
    //   int neiTetIdx = oldOpp.getOppTet( vi );
    //   int neiTetVi  = oldOpp.getOppVi( vi );
    //   const int neiSplitIdx = tetToVert[ neiTetIdx ];
    //   if ( neiSplitIdx != INT_MAX ) {
    //       neiTetIdx = freeArr[ neiFreeIdx - neiTetVi ];
    //       neiTetVi  = 3;
    //   }
    //   newOpp.setOpp( 3, neiTetIdx, neiTetVi );
    //   oppArr[ neiTetIdx ].setOpp( neiTetVi, newTetIdx[vi], 3 );
    //
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

            // ═════════════════════════════════════════════════════════════════════
            // CRITICAL FIX: Set OUTGOING edge (this tet's face 3 → neighbor)
            // ═════════════════════════════════════════════════════════════════════
            // This was missing! Without it, face 3 points to old dead tets.
            set_opp_at(new_tets[k], 3u, encode_opp(nei_tet, nei_face));

            // Set INCOMING edge (neighbor → this tet's face 3)
            set_opp_at(nei_tet, nei_face, encode_opp(new_tets[k], 3u));
        } else {
            // No external neighbor (boundary face) - mark as invalid
            set_opp_at(new_tets[k], 3u, INVALID);
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
