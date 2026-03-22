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

// Encode internal adjacency (between split sibling tets).
// Sets bit 2 = internal flag so check_delaunay_fast skips these faces.
// CUDA: CommonTypes.h:277-279 setOppInternal
fn encode_opp_internal(tet_idx: u32, face: u32) -> u32 {
    return (tet_idx << 5u) | 4u | (face & 3u);
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

// Allocate 4 consecutive tet slots deterministically based on insertion index.
// Each insertion position `idx` gets tets at [base + idx*4, ..., base + idx*4 + 3]
// where base = params.w (next_free_tet, provided by CPU).
//
// Deterministic allocation is REQUIRED so that split_points.wgsl (which runs
// BEFORE split) can predict the new tet indices using the same formula.
fn get_free_slots_4tet(idx: u32) -> vec4<u32> {
    let base = params.w + idx * 4u;
    return vec4<u32>(base, base + 1u, base + 2u, base + 3u);
}

@compute @workgroup_size(256)
fn split_tetra(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let tid = gid.x;  // Thread ID for debugging
    let idx = tid;

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

    // NOTE: Removed degenerate tet check - CUDA has no such validation
    // If degenerate tets exist, they indicate bugs elsewhere in the pipeline

    // Get the actual vertex ID (p is position in uninserted array, not vertex ID!)
    let vertex = uninserted[p];

    // NOTE: Removed duplicate vertex check - CUDA has no such validation

    // Allocate 4 new tet slots deterministically based on insertion index
    let new_slots = get_free_slots_4tet(idx);
    let t0 = new_slots.x;
    let t1 = new_slots.y;
    let t2 = new_slots.z;
    let t3 = new_slots.w;

    breadcrumb(tid, CRUMB_AFTER_ALLOC);
    debug_slot(tid, 1u, new_slots);

    // NOTE: Removed allocation failure check - CUDA has no such validation
    // Pre-allocation should ensure enough slots are always available

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

    // Memory barrier: Ensure vertex writes are visible before TET_ALIVE flag is set
    // (Prevents race where another thread reads TET_ALIVE=1 with stale vertices)
    storageBarrier();

    breadcrumb(tid, CRUMB_AFTER_WRITE_ALL);

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
    set_opp_at(t0, 0u, encode_opp_internal(t1, 0u));   // face 0: T1's face 0 (internal)
    set_opp_at(t0, 1u, encode_opp_internal(t3, 0u));   // face 1: T3's face 0 (internal)
    set_opp_at(t0, 2u, encode_opp_internal(t2, 0u));   // face 2: T2's face 0 (internal)
    // face 3: external - set below after detecting concurrent splits

    // T1 (vi=1) adjacency: IntSplitFaceOpp[1] = {0, 0, 2, 2, 3, 1}
    set_opp_at(t1, 0u, encode_opp_internal(t0, 0u));   // face 0: T0's face 0 (internal)
    set_opp_at(t1, 1u, encode_opp_internal(t2, 2u));   // face 1: T2's face 2 (internal)
    set_opp_at(t1, 2u, encode_opp_internal(t3, 1u));   // face 2: T3's face 1 (internal)
    // face 3: external - set below after detecting concurrent splits

    // T2 (vi=2) adjacency: IntSplitFaceOpp[2] = {0, 2, 3, 2, 1, 1}
    set_opp_at(t2, 0u, encode_opp_internal(t0, 2u));   // face 0: T0's face 2 (internal)
    set_opp_at(t2, 1u, encode_opp_internal(t3, 2u));   // face 1: T3's face 2 (internal)
    set_opp_at(t2, 2u, encode_opp_internal(t1, 1u));   // face 2: T1's face 1 (internal)
    // face 3: external - set below after detecting concurrent splits

    // T3 (vi=3) adjacency: IntSplitFaceOpp[3] = {0, 1, 1, 2, 2, 1}
    set_opp_at(t3, 0u, encode_opp_internal(t0, 1u));   // face 0: T0's face 1 (internal)
    set_opp_at(t3, 1u, encode_opp_internal(t1, 2u));   // face 1: T1's face 2 (internal)
    set_opp_at(t3, 2u, encode_opp_internal(t2, 1u));   // face 2: T2's face 1 (internal)
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
            // tet_to_vert[nei_tet] is pre-written by CPU with the base tet index
            // for all tets being split this iteration. INVALID = not being split.
            let nei_base_tet = tet_to_vert[nei_tet];

            if nei_base_tet != INVALID {
                // Neighbor is being split concurrently.
                // Its 4 new sub-tets are at [nei_base_tet, nei_base_tet+1, +2, +3].
                // Sub-tet vi has face 3 = the external face that was original face vi.
                // So the sub-tet we need is nei_base_tet + nei_face.
                // CUDA ref: KerDivision.cu:155-159
                nei_tet = nei_base_tet + nei_face;
                nei_face = 3u;  // External faces become face 3 after split
            } else {
                // Neighbor is NOT split - update its incoming edge to point to us.
                // CUDA ref: KerDivision.cu:150
                // Only done for unsplit neighbors! If neighbor IS split, its own
                // thread will set the incoming edge when it processes face 3.
                set_opp_at(nei_tet, nei_face, encode_opp(new_tets[k], 3u));
            }

            // Set OUTGOING edge (this tet's face 3 → neighbor or neighbor's sub-tet)
            set_opp_at(new_tets[k], 3u, encode_opp(nei_tet, nei_face));
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

    // tet_to_vert[old_tet] already contains t0 (base tet), written by CPU
    // before the split dispatch. split_points reads this to find the 4 new tets.
    // No write needed here — the CPU-written value is correct and already visible.

    // ═══════════════════════════════════════════════════════════════════════════
    // Mark old tet as dead (but do NOT donate back to free list)
    // ═══════════════════════════════════════════════════════════════════════════
    // CUDA donates old tet to its block owner's free list (KerDivision.cu:181-188).
    // However, in our WGPU port, donation races with allocation: concurrent
    // atomicAdd (donation) and atomicSub (allocation) on vert_free_arr can
    // interleave, causing the counter to overflow the block boundary. The
    // allocation then reads tet indices from an adjacent block, which may be
    // alive tets → corruption.
    //
    // Fix: just mark dead. We pre-allocate (num_points+5)*8 tets, so each
    // vertex has 4 spare slots after split. Flip allocation falls back to
    // inf_idx pool if a vertex's local pool is exhausted.
    tet_info[old_tet] = 0u;  // Mark as dead (slot is wasted, not recycled)
    // ═══════════════════════════════════════════════════════════════════════════

    breadcrumb(tid, CRUMB_AFTER_MARK_DEAD);

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
