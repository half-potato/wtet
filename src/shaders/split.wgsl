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
// Dispatch: ceil(num_insertions / 64)

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> tets: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> tet_opp: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> tet_info: array<u32>;
@group(0) @binding(4) var<storage, read_write> vert_tet: array<u32>;
@group(0) @binding(5) var<storage, read> insert_list: array<vec2<u32>>; // (tet_idx, vert_idx)
@group(0) @binding(6) var<storage, read_write> free_stack: array<u32>;
@group(0) @binding(7) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> flip_queue: array<u32>; // tets needing flip check
@group(0) @binding(9) var<uniform> params: vec4<u32>; // x = num_insertions

const INVALID: u32 = 0xFFFFFFFFu;
const TET_ALIVE: u32 = 1u;
const TET_CHANGED: u32 = 2u;
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

// Pop 3 free slots from the free stack.
fn pop_free_slots(count: u32) -> vec3<u32> {
    let old_free = atomicSub(&counters[COUNTER_FREE], count);
    var slots: vec3<u32>;
    slots.x = free_stack[old_free - 1u];
    slots.y = free_stack[old_free - 2u];
    slots.z = free_stack[old_free - 3u];
    return slots;
}

@compute @workgroup_size(64)
fn split_tetra(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_insertions = params.x;

    if idx >= num_insertions {
        return;
    }

    let insert = insert_list[idx];
    let t0 = insert.x; // original tet index
    let p = insert.y;  // point to insert

    // Read original tet
    let orig = tets[t0];
    let orig_opp = tet_opp[t0];
    let v0 = orig.x;
    let v1 = orig.y;
    let v2 = orig.z;
    let v3 = orig.w;

    // Allocate 3 new tet slots
    let new_slots = pop_free_slots(3u);
    let t1 = new_slots.x;
    let t2 = new_slots.y;
    let t3 = new_slots.z;

    // Write the 4 new tets:
    // T0 = (P, v1, v2, v3) — replaces v0 with P
    tets[t0] = vec4<u32>(p, v1, v2, v3);
    // T1 = (v0, P, v2, v3) — replaces v1 with P
    tets[t1] = vec4<u32>(v0, p, v2, v3);
    // T2 = (v0, v1, P, v3) — replaces v2 with P
    tets[t2] = vec4<u32>(v0, v1, p, v3);
    // T3 = (v0, v1, v2, P) — replaces v3 with P
    tets[t3] = vec4<u32>(v0, v1, v2, p);

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
    tet_opp[t0] = vec4<u32>(
        orig_opp.x,           // face 0: external (was opp v0)
        encode_opp(t1, 0u),   // face 1: T1's face 0
        encode_opp(t2, 0u),   // face 2: T2's face 0
        encode_opp(t3, 0u)    // face 3: T3's face 0
    );

    // T1 adjacency:
    tet_opp[t1] = vec4<u32>(
        encode_opp(t0, 1u),   // face 0: T0's face 1
        orig_opp.y,           // face 1: external (was opp v1)
        encode_opp(t2, 1u),   // face 2: T2's face 1
        encode_opp(t3, 1u)    // face 3: T3's face 1
    );

    // T2 adjacency:
    tet_opp[t2] = vec4<u32>(
        encode_opp(t0, 2u),   // face 0: T0's face 2
        encode_opp(t1, 2u),   // face 1: T1's face 2
        orig_opp.z,           // face 2: external (was opp v2)
        encode_opp(t3, 2u)    // face 3: T3's face 2
    );

    // T3 adjacency:
    tet_opp[t3] = vec4<u32>(
        encode_opp(t0, 3u),   // face 0: T0's face 3
        encode_opp(t1, 3u),   // face 1: T1's face 3
        encode_opp(t2, 3u),   // face 2: T2's face 3
        orig_opp.w            // face 3: external (was opp v3)
    );

    // Update external neighbours to point back to us.
    // For each face k, the external neighbour (if any) that pointed to (t0, face k)
    // now needs to point to (tk, face k).
    for (var k = 0u; k < 4u; k++) {
        var ext_opp: u32;
        switch k {
            case 0u: { ext_opp = orig_opp.x; }
            case 1u: { ext_opp = orig_opp.y; }
            case 2u: { ext_opp = orig_opp.z; }
            default: { ext_opp = orig_opp.w; }
        }

        if ext_opp != INVALID {
            let ext_tet = decode_opp_tet(ext_opp);
            let ext_face = decode_opp_face(ext_opp);

            var new_tet: u32;
            switch k {
                case 0u: { new_tet = t0; }
                case 1u: { new_tet = t1; }
                case 2u: { new_tet = t2; }
                default: { new_tet = t3; }
            }

            // Update the external neighbour's opp entry
            // tet_opp[ext_tet][ext_face] = encode_opp(new_tet, k)
            var opp_val = tet_opp[ext_tet];
            switch ext_face {
                case 0u: { opp_val.x = encode_opp(new_tet, k); }
                case 1u: { opp_val.y = encode_opp(new_tet, k); }
                case 2u: { opp_val.z = encode_opp(new_tet, k); }
                default: { opp_val.w = encode_opp(new_tet, k); }
            }
            tet_opp[ext_tet] = opp_val;
        }
    }

    // Mark all 4 tets as alive + changed (need flip checking)
    tet_info[t0] = TET_ALIVE | TET_CHANGED;
    tet_info[t1] = TET_ALIVE | TET_CHANGED;
    tet_info[t2] = TET_ALIVE | TET_CHANGED;
    tet_info[t3] = TET_ALIVE | TET_CHANGED;

    // Update active count (added 3, original already counted)
    atomicAdd(&counters[COUNTER_ACTIVE], 3u);

    // Map the inserted vertex to one of its tets
    vert_tet[p] = t0;

    // Add all 4 tets to flip queue (they need Delaunay checking)
    let fq_base = idx * 4u;
    flip_queue[fq_base] = t0;
    flip_queue[fq_base + 1u] = t1;
    flip_queue[fq_base + 2u] = t2;
    flip_queue[fq_base + 3u] = t3;
}
