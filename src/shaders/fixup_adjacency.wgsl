// Kernel: Fix broken back-pointers after colored splitting.
//
// After 4-color splits complete, some tets may have correct forward pointers
// but broken back-pointers (neighbors pointing to old deleted tets).
// This pass scans active tets and fixes any broken back-pointers.

@group(0) @binding(0) var<storage, read> tets: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> tet_opp: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> tet_info: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x = max_tets

const INVALID: u32 = 0xFFFFFFFFu;
const TET_ALIVE: u32 = 1u;

// TetOpp encoding from CommonTypes.h line 266 - must match init.wgsl and split.wgsl
fn encode_opp(tet_idx: u32, opp_vi: u32) -> u32 {
    return (tet_idx << 5u) | opp_vi;
}

fn decode_opp_tet(packed: u32) -> u32 {
    return packed >> 5u;
}

fn decode_opp_vi(packed: u32) -> u32 {
    return packed & 31u; // Extract lower 5 bits
}

fn get_opp(tet_idx: u32, face: u32) -> u32 {
    return atomicLoad(&tet_opp[tet_idx * 4u + face]);
}

fn set_opp_at(tet_idx: u32, face: u32, val: u32) {
    atomicStore(&tet_opp[tet_idx * 4u + face], val);
}

@compute @workgroup_size(64)
fn fixup_adjacency(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let tet_idx = gid.x;
    let max_tets = params.x;

    if tet_idx >= max_tets {
        return;
    }

    // Only process alive tets
    if (tet_info[tet_idx] & TET_ALIVE) == 0u {
        return;
    }

    // Check each face
    for (var face = 0u; face < 4u; face++) {
        let my_opp = get_opp(tet_idx, face);

        if my_opp != INVALID {
            let nei_tet = decode_opp_tet(my_opp);
            let nei_face = decode_opp_vi(my_opp);

            // Bounds check
            if nei_tet >= max_tets {
                continue;
            }

            // Check if neighbor is alive
            if (tet_info[nei_tet] & TET_ALIVE) == 0u {
                // Neighbor is dead, clear our pointer
                set_opp_at(tet_idx, face, INVALID);
                continue;
            }

            // Check if neighbor points back to us
            let nei_back_opp = get_opp(nei_tet, nei_face);
            let back_tet = decode_opp_tet(nei_back_opp);
            let back_face = decode_opp_vi(nei_back_opp);

            // If neighbor doesn't point back, fix it
            if back_tet != tet_idx || back_face != face {
                set_opp_at(nei_tet, nei_face, encode_opp(tet_idx, face));
            }
        }
    }
}
