// Port of kerSplitTetra from gDel3D/GPU/KerDivision.cu (lines 98-192)
// This is a direct translation maintaining the same algorithm and structure

@group(0) @binding(0) var<storage, read_write> tets: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> tet_opp: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> tet_info: array<u32>;
@group(0) @binding(3) var<storage, read_write> vert_tet: array<u32>;
@group(0) @binding(4) var<storage, read> insert_list: array<vec2<u32>>; // (tet_idx, position)
@group(0) @binding(5) var<storage, read_write> free_arr: array<u32>;
@group(0) @binding(6) var<storage, read_write> vert_free_arr: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> flip_queue: array<u32>;
@group(0) @binding(9) var<storage, read_write> tet_to_vert: array<u32>;
@group(0) @binding(10) var<uniform> params: vec4<u32>; // x = num_insertions
@group(0) @binding(11) var<storage, read_write> breadcrumbs: array<u32>;
@group(0) @binding(12) var<storage, read_write> thread_debug: array<vec4<u32>>;
@group(0) @binding(13) var<storage, read> uninserted: array<u32>;

const INVALID: u32 = 0xFFFFFFFFu;
const INT_MAX: u32 = 0xFFFFFFFFu;
const TET_ALIVE: u32 = 1u;
const TET_CHANGED: u32 = 2u;
const COUNTER_ACTIVE: u32 = 1u;
const MEAN_VERTEX_DEGREE: u32 = 8u; // From original KerCommon.h line 56

// From original CommonTypes.h line 128
const TET_VI_AS_SEEN_FROM: array<array<u32, 3>, 4> = array<array<u32, 3>, 4>(
    array<u32, 3>(1u, 3u, 2u), // From 0
    array<u32, 3>(0u, 2u, 3u), // From 1
    array<u32, 3>(0u, 3u, 1u), // From 2
    array<u32, 3>(0u, 1u, 2u), // From 3
);

// From original GPUDecl.h line 162
const INT_SPLIT_FACE_OPP: array<array<u32, 6>, 4> = array<array<u32, 6>, 4>(
    array<u32, 6>(1u, 0u, 3u, 0u, 2u, 0u),
    array<u32, 6>(0u, 0u, 2u, 2u, 3u, 1u),
    array<u32, 6>(0u, 2u, 3u, 2u, 1u, 1u),
    array<u32, 6>(0u, 1u, 1u, 2u, 2u, 1u),
);

// Helper functions for TetOpp encoding (from original CommonTypes.h lines 266-268)
fn make_opp_val(tet_idx: u32, opp_tet_vi: u32) -> u32 {
    return (tet_idx << 5u) | opp_tet_vi;
}

fn get_opp_val_tet(val: u32) -> u32 {
    return val >> 5u;
}

fn get_opp_val_vi(val: u32) -> u32 {
    return val & 3u;
}

// Load/store functions (from original KerCommon.h lines 111-143)
fn load_tet(idx: u32) -> vec4<u32> {
    return tets[idx];
}

fn store_tet(idx: u32, tet: vec4<u32>) {
    tets[idx] = tet;
}

fn load_opp(idx: u32) -> vec4<u32> {
    return vec4<u32>(
        atomicLoad(&tet_opp[idx * 4u + 0u]),
        atomicLoad(&tet_opp[idx * 4u + 1u]),
        atomicLoad(&tet_opp[idx * 4u + 2u]),
        atomicLoad(&tet_opp[idx * 4u + 3u]),
    );
}

fn store_opp(idx: u32, opp: vec4<u32>) {
    atomicStore(&tet_opp[idx * 4u + 0u], opp.x);
    atomicStore(&tet_opp[idx * 4u + 1u], opp.y);
    atomicStore(&tet_opp[idx * 4u + 2u], opp.z);
    atomicStore(&tet_opp[idx * 4u + 3u], opp.w);
}

fn set_opp_at(tet_idx: u32, vi: u32, val: u32) {
    atomicStore(&tet_opp[tet_idx * 4u + vi], val);
}

// From original CommonTypes.h line 275-283 (setOpp and setOppInternal)
fn set_opp_internal_bit(val: u32) -> u32 {
    return val | (1u << 2u); // Set bit 2 for internal flag
}

// Direct port of kerSplitTetra (lines 98-192)
@compute @workgroup_size(64)
fn split_tetra(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_insertions = params.x;

    // Line 114: for ( int idx = getCurThreadIdx(); idx < newVertVec._num; idx += getThreadNum() )
    if idx >= num_insertions {
        return;
    }

    // Lines 116-121: Get insertion info and allocate new tets
    // In original: insIdx = newVertVec._arr[idx], tetIdx = vertTetArr[insIdx], splitVertex = vertArr[insIdx]
    // insert_list[idx] = (tetIdx, position) where position is idx in uninserted array
    let insert = insert_list[idx];
    let tet_idx = insert.x;
    let uninserted_pos = insert.y;
    let split_vertex = uninserted[uninserted_pos]; // Get actual vertex index

    let new_idx = (split_vertex + 1u) * MEAN_VERTEX_DEGREE - 1u;
    let new_tet_idx_0 = free_arr[new_idx];
    let new_tet_idx_1 = free_arr[new_idx - 1u];
    let new_tet_idx_2 = free_arr[new_idx - 2u];
    let new_tet_idx_3 = free_arr[new_idx - 3u];

    // Line 124: Update vertFree, 4 has been used
    atomicSub(&vert_free_arr[split_vertex], 4u);

    // Lines 127-128: Load old tet and opp
    let old_opp = load_opp(tet_idx);
    let tet = load_tet(tet_idx);

    // Lines 130-179: For each vi = 0..3, create new tet
    // UNROLLED MANUALLY to avoid WGSL for-loop issues

    // === vi = 0 ===
    {
        let vi = 0u;
        var new_opp = vec4<u32>(INVALID, INVALID, INVALID, INVALID);

        // Lines 135-137: Set internal adjacency
        new_opp.x = set_opp_internal_bit(make_opp_val(new_tet_idx_1, INT_SPLIT_FACE_OPP[vi][1]));
        new_opp.y = set_opp_internal_bit(make_opp_val(new_tet_idx_3, INT_SPLIT_FACE_OPP[vi][3]));
        new_opp.z = set_opp_internal_bit(make_opp_val(new_tet_idx_2, INT_SPLIT_FACE_OPP[vi][5]));

        // Lines 142-163: Set external adjacency
        var nei_tet_idx = get_opp_val_tet(old_opp[vi]);
        var nei_tet_vi = get_opp_val_vi(old_opp[vi]);

        // Line 146: Check if neighbour has split
        let nei_split_idx = tet_to_vert[nei_tet_idx];

        if nei_split_idx == INT_MAX {
            // Lines 148-150: Neighbour is un-split
            set_opp_at(nei_tet_idx, nei_tet_vi, make_opp_val(new_tet_idx_0, 3u));
        } else {
            // Lines 152-160: Neighbour has split
            let nei_split_pos = insert_list[nei_split_idx].y;
            let nei_split_vert = uninserted[nei_split_pos];
            let nei_free_idx = (nei_split_vert + 1u) * MEAN_VERTEX_DEGREE - 1u;
            nei_tet_idx = free_arr[nei_free_idx - nei_tet_vi];
            nei_tet_vi = 3u;
        }

        // Line 162: Point this tetra to neighbour
        new_opp.w = make_opp_val(nei_tet_idx, nei_tet_vi);

        // Lines 166-171: Create new tet
        let new_tet = vec4<u32>(
            tet[TET_VI_AS_SEEN_FROM[vi][0]],
            tet[TET_VI_AS_SEEN_FROM[vi][1]],
            tet[TET_VI_AS_SEEN_FROM[vi][2]],
            split_vertex
        );

        // Lines 174-178: Store tet, opp, and set flags
        store_tet(new_tet_idx_0, new_tet);
        store_opp(new_tet_idx_0, new_opp);
        tet_info[new_tet_idx_0] = TET_ALIVE | TET_CHANGED;
    }

    // === vi = 1 ===
    {
        let vi = 1u;
        var new_opp = vec4<u32>(INVALID, INVALID, INVALID, INVALID);

        new_opp.x = set_opp_internal_bit(make_opp_val(new_tet_idx_0, INT_SPLIT_FACE_OPP[vi][1]));
        new_opp.y = set_opp_internal_bit(make_opp_val(new_tet_idx_2, INT_SPLIT_FACE_OPP[vi][3]));
        new_opp.z = set_opp_internal_bit(make_opp_val(new_tet_idx_3, INT_SPLIT_FACE_OPP[vi][5]));

        var nei_tet_idx = get_opp_val_tet(old_opp[vi]);
        var nei_tet_vi = get_opp_val_vi(old_opp[vi]);

        let nei_split_idx = tet_to_vert[nei_tet_idx];

        if nei_split_idx == INT_MAX {
            set_opp_at(nei_tet_idx, nei_tet_vi, make_opp_val(new_tet_idx_1, 3u));
        } else {
            let nei_split_pos = insert_list[nei_split_idx].y;
            let nei_split_vert = uninserted[nei_split_pos];
            let nei_free_idx = (nei_split_vert + 1u) * MEAN_VERTEX_DEGREE - 1u;
            nei_tet_idx = free_arr[nei_free_idx - nei_tet_vi];
            nei_tet_vi = 3u;
        }

        new_opp.w = make_opp_val(nei_tet_idx, nei_tet_vi);

        let new_tet = vec4<u32>(
            tet[TET_VI_AS_SEEN_FROM[vi][0]],
            tet[TET_VI_AS_SEEN_FROM[vi][1]],
            tet[TET_VI_AS_SEEN_FROM[vi][2]],
            split_vertex
        );

        store_tet(new_tet_idx_1, new_tet);
        store_opp(new_tet_idx_1, new_opp);
        tet_info[new_tet_idx_1] = TET_ALIVE | TET_CHANGED;
    }

    // === vi = 2 ===
    {
        let vi = 2u;
        var new_opp = vec4<u32>(INVALID, INVALID, INVALID, INVALID);

        new_opp.x = set_opp_internal_bit(make_opp_val(new_tet_idx_0, INT_SPLIT_FACE_OPP[vi][1]));
        new_opp.y = set_opp_internal_bit(make_opp_val(new_tet_idx_3, INT_SPLIT_FACE_OPP[vi][3]));
        new_opp.z = set_opp_internal_bit(make_opp_val(new_tet_idx_1, INT_SPLIT_FACE_OPP[vi][5]));

        var nei_tet_idx = get_opp_val_tet(old_opp[vi]);
        var nei_tet_vi = get_opp_val_vi(old_opp[vi]);

        let nei_split_idx = tet_to_vert[nei_tet_idx];

        if nei_split_idx == INT_MAX {
            set_opp_at(nei_tet_idx, nei_tet_vi, make_opp_val(new_tet_idx_2, 3u));
        } else {
            let nei_split_pos = insert_list[nei_split_idx].y;
            let nei_split_vert = uninserted[nei_split_pos];
            let nei_free_idx = (nei_split_vert + 1u) * MEAN_VERTEX_DEGREE - 1u;
            nei_tet_idx = free_arr[nei_free_idx - nei_tet_vi];
            nei_tet_vi = 3u;
        }

        new_opp.w = make_opp_val(nei_tet_idx, nei_tet_vi);

        let new_tet = vec4<u32>(
            tet[TET_VI_AS_SEEN_FROM[vi][0]],
            tet[TET_VI_AS_SEEN_FROM[vi][1]],
            tet[TET_VI_AS_SEEN_FROM[vi][2]],
            split_vertex
        );

        store_tet(new_tet_idx_2, new_tet);
        store_opp(new_tet_idx_2, new_opp);
        tet_info[new_tet_idx_2] = TET_ALIVE | TET_CHANGED;
    }

    // === vi = 3 ===
    {
        let vi = 3u;
        var new_opp = vec4<u32>(INVALID, INVALID, INVALID, INVALID);

        new_opp.x = set_opp_internal_bit(make_opp_val(new_tet_idx_0, INT_SPLIT_FACE_OPP[vi][1]));
        new_opp.y = set_opp_internal_bit(make_opp_val(new_tet_idx_1, INT_SPLIT_FACE_OPP[vi][3]));
        new_opp.z = set_opp_internal_bit(make_opp_val(new_tet_idx_2, INT_SPLIT_FACE_OPP[vi][5]));

        var nei_tet_idx = get_opp_val_tet(old_opp[vi]);
        var nei_tet_vi = get_opp_val_vi(old_opp[vi]);

        let nei_split_idx = tet_to_vert[nei_tet_idx];

        if nei_split_idx == INT_MAX {
            set_opp_at(nei_tet_idx, nei_tet_vi, make_opp_val(new_tet_idx_3, 3u));
        } else {
            let nei_split_pos = insert_list[nei_split_idx].y;
            let nei_split_vert = uninserted[nei_split_pos];
            let nei_free_idx = (nei_split_vert + 1u) * MEAN_VERTEX_DEGREE - 1u;
            nei_tet_idx = free_arr[nei_free_idx - nei_tet_vi];
            nei_tet_vi = 3u;
        }

        new_opp.w = make_opp_val(nei_tet_idx, nei_tet_vi);

        let new_tet = vec4<u32>(
            tet[TET_VI_AS_SEEN_FROM[vi][0]],
            tet[TET_VI_AS_SEEN_FROM[vi][1]],
            tet[TET_VI_AS_SEEN_FROM[vi][2]],
            split_vertex
        );

        store_tet(new_tet_idx_3, new_tet);
        store_opp(new_tet_idx_3, new_opp);
        tet_info[new_tet_idx_3] = TET_ALIVE | TET_CHANGED;
    }

    // Lines 181-188: Donate one tetra back to free list
    // Note: In original, this uses insVertVec which we don't have in same form
    // Simplified version - donate back to the vertex that's being inserted
    let free_idx_ret = atomicAdd(&vert_free_arr[split_vertex], 1u);
    free_arr[split_vertex * MEAN_VERTEX_DEGREE + free_idx_ret] = tet_idx;

    // Line 188: Mark old tet as dead
    tet_info[tet_idx] = 0u;

    // Update active count (+4 new, -1 old = net +3)
    atomicAdd(&counters[COUNTER_ACTIVE], 3u);

    // Map the inserted vertex to one of its tets (use position, not vertex!)
    vert_tet[uninserted_pos] = new_tet_idx_0;

    // Add all 4 tets to flip queue
    let fq_base = idx * 4u;
    flip_queue[fq_base] = new_tet_idx_0;
    flip_queue[fq_base + 1u] = new_tet_idx_1;
    flip_queue[fq_base + 2u] = new_tet_idx_2;
    flip_queue[fq_base + 3u] = new_tet_idx_3;

    // Clear tet_to_vert for newly allocated tets
    tet_to_vert[new_tet_idx_0] = INVALID;
    tet_to_vert[new_tet_idx_1] = INVALID;
    tet_to_vert[new_tet_idx_2] = INVALID;
    tet_to_vert[new_tet_idx_3] = INVALID;
}
