// Post-flip adjacency repair pass.
//
// After flip.wgsl executes, forward pointers (this_tet → neighbor) have the
// correct SLOT but potentially stale FACE INDEX (if the neighbor was also
// flipped concurrently and its face assignment changed).
//
// Additionally, external neighbors' back-pointers (neighbor → this_tet) are
// stale because flip.wgsl does NOT write back-pointers (removed to avoid
// race conditions during concurrent flips).
//
// This shader fixes both issues using vertex matching:
//   1. For each alive tet, verify each non-internal forward pointer
//   2. Find the correct face of the neighbor by matching face vertices
//   3. Fix own forward pointer (face index correction)
//   4. Write back-pointer to neighbor (fixes their stale forward pointer)
//   5. If no match on direct neighbor, 1-hop search: check neighbor's neighbors
//      (handles concurrent flips that moved the shared face to a new tet)
//
// Race-free because:
//   - Each face of a tet has exactly one neighbor, so no two threads
//     write to the same (tet, face) entry
//   - Back-pointer writes are safe: the neighbor's face is uniquely
//     determined by vertex matching
//
// Bindings:
// - @binding(0): tets (read) - per-tet vertex indices
// - @binding(1): tet_opp (read_write) - per-tet adjacency (4 entries per tet)
// - @binding(2): tet_info (read) - per-tet flags (alive/changed)
// - @binding(3): params (uniform) - x = total_tet_num

@group(0) @binding(0) var<storage, read> tets: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> tet_opp: array<u32>;
@group(0) @binding(2) var<storage, read> tet_info: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

const INVALID: u32 = 0xFFFFFFFFu;
const TET_ALIVE: u32 = 1u;
const OPP_INTERNAL: u32 = 4u;

fn decode_opp_tet(packed: u32) -> u32 { return packed >> 5u; }
fn decode_opp_face(packed: u32) -> u32 { return packed & 3u; }
fn encode_opp(tet_idx: u32, face: u32) -> u32 { return (tet_idx << 5u) | (face & 3u); }

// Get the 3 face vertices of a tet (the vertices NOT at position `face`).
// Returns them in a consistent order for matching.
fn get_face_verts(tet: vec4<u32>, face: u32) -> vec3<u32> {
    if face == 0u { return vec3<u32>(tet.y, tet.z, tet.w); }
    if face == 1u { return vec3<u32>(tet.x, tet.z, tet.w); }
    if face == 2u { return vec3<u32>(tet.x, tet.y, tet.w); }
    return vec3<u32>(tet.x, tet.y, tet.z); // face == 3
}

// Check if vertex v is one of the 3 face vertices.
fn face_contains(fv: vec3<u32>, v: u32) -> bool {
    return fv.x == v || fv.y == v || fv.z == v;
}

// Check if two faces share the same 3 vertices (order-independent).
fn faces_match(a: vec3<u32>, b: vec3<u32>) -> bool {
    return face_contains(b, a.x) && face_contains(b, a.y) && face_contains(b, a.z);
}

@compute @workgroup_size(256)
fn repair_adjacency(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.x { return; }

    let info = tet_info[idx];
    if (info & TET_ALIVE) == 0u { return; }

    let tet = tets[idx];

    for (var f = 0u; f < 4u; f++) {
        let opp = tet_opp[idx * 4u + f];
        if opp == INVALID { continue; }

        // Skip internal faces — set by same flip, always correct
        if (opp & OPP_INTERNAL) != 0u { continue; }

        let nei_slot = decode_opp_tet(opp);
        let nei_face_expected = decode_opp_face(opp);

        // Check neighbor is alive
        let nei_info = tet_info[nei_slot];
        if (nei_info & TET_ALIVE) == 0u {
            // Neighbor is dead — forward pointer is stale
            tet_opp[idx * 4u + f] = INVALID;
            continue;
        }

        let nei_tet = tets[nei_slot];
        let my_face_verts = get_face_verts(tet, f);

        // Fast path: check if expected face is correct
        let expected_verts = get_face_verts(nei_tet, nei_face_expected);
        if faces_match(my_face_verts, expected_verts) {
            // Forward pointer is correct. Write back-pointer to fix neighbor.
            tet_opp[nei_slot * 4u + nei_face_expected] = encode_opp(idx, f);
            continue;
        }

        // Slow path: search all 4 faces of neighbor for matching face
        var found = false;
        for (var nf = 0u; nf < 4u; nf++) {
            if nf == nei_face_expected { continue; } // Already checked
            let nf_verts = get_face_verts(nei_tet, nf);
            if faces_match(my_face_verts, nf_verts) {
                // Fix own forward pointer (face index was wrong)
                tet_opp[idx * 4u + f] = encode_opp(nei_slot, nf);
                // Write back-pointer to neighbor
                tet_opp[nei_slot * 4u + nf] = encode_opp(idx, f);
                found = true;
                break;
            }
        }

        if found { continue; }

        // 1-hop search: the neighbor was concurrently flipped and the shared
        // face migrated to a different tet slot (e.g., a new tet created by
        // the concurrent flip). Search the neighbor's neighbors.
        for (var nf = 0u; nf < 4u; nf++) {
            let hop_opp = tet_opp[nei_slot * 4u + nf];
            if hop_opp == INVALID { continue; }

            let hop_slot = decode_opp_tet(hop_opp);
            if hop_slot == idx { continue; } // Skip self
            if hop_slot == nei_slot { continue; } // Skip self-loop

            let hop_info = tet_info[hop_slot];
            if (hop_info & TET_ALIVE) == 0u { continue; }

            let hop_tet = tets[hop_slot];

            for (var hf = 0u; hf < 4u; hf++) {
                let hf_verts = get_face_verts(hop_tet, hf);
                if faces_match(my_face_verts, hf_verts) {
                    // Found the face on a neighbor's neighbor!
                    // Fix own forward pointer to point to the correct tet+face
                    tet_opp[idx * 4u + f] = encode_opp(hop_slot, hf);
                    // Write back-pointer
                    tet_opp[hop_slot * 4u + hf] = encode_opp(idx, f);
                    found = true;
                    break;
                }
            }
            if found { break; }
        }

        if !found {
            // Face truly lost — invalidate to prevent walking into stale data.
            tet_opp[idx * 4u + f] = INVALID;
        }
    }
}
