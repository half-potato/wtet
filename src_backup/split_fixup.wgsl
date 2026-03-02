// Kernel: Fix up adjacency for tets whose neighbors also split
//
// After the split kernel runs, tets whose external neighbors were also splitting
// have stale adjacency (pointing to old tet indices). This kernel fixes them up
// by looking at the tet_split_map to find the new tet indices.
//
// For each insertion, we need to check all 4 new tets' external faces.

@group(0) @binding(0) var<storage, read> insert_list: array<vec2<u32>>; // (tet_idx, vert_idx)
@group(0) @binding(1) var<storage, read> tet_to_vert: array<u32>; // maps tet_idx -> vert (or INVALID)
@group(0) @binding(2) var<storage, read> tet_split_map: array<vec4<u32>>; // old_tet -> (t0,t1,t2,t3)
@group(0) @binding(3) var<storage, read_write> tet_opp: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: vec4<u32>; // x = num_insertions

const INVALID: u32 = 0xFFFFFFFFu;

fn decode_opp_tet(packed: u32) -> u32 {
    return packed >> 2u;
}

fn decode_opp_face(packed: u32) -> u32 {
    return packed & 3u;
}

fn encode_opp(tet_idx: u32, face: u32) -> u32 {
    return (tet_idx << 2u) | (face & 3u);
}

fn get_opp(tet_idx: u32, face: u32) -> u32 {
    return atomicLoad(&tet_opp[tet_idx * 4u + face]);
}

fn set_opp_at(tet_idx: u32, face: u32, val: u32) {
    atomicStore(&tet_opp[tet_idx * 4u + face], val);
}

@compute @workgroup_size(64)
fn fixup_split_adjacency(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_insertions = params.x;

    if idx >= num_insertions {
        return;
    }

    let insert = insert_list[idx];
    let old_tet = insert.x;

    // Get the 4 new tets from the split mapping
    let new_tets = tet_split_map[old_tet];

    // For each of the 4 new tets, check their external faces
    // Following CUDA convention: face 3 is always the external face for all new tets
    // new_tets[k] corresponds to original face k, which is now at face 3 of the new tet
    for (var k = 0u; k < 4u; k++) {
        let my_tet = new_tets[k];
        let my_face = 3u;  // External face is always face 3
        let opp_packed = get_opp(my_tet, my_face);

        if opp_packed == INVALID {
            continue;
        }

        let nei_tet = decode_opp_tet(opp_packed);
        let nei_face = decode_opp_face(opp_packed);

        // Bounds check: if nei_tet is out of range, skip
        if nei_tet >= arrayLength(&tet_to_vert) {
            continue;
        }

        // Check if the neighbor was also split
        let nei_vert = tet_to_vert[nei_tet];
        if nei_vert != INVALID {
            // Neighbor was split! Look up its new tets
            let nei_new_tets = tet_split_map[nei_tet];

            // The neighbor's original face nei_face is now at face 3 of nei_new_tets[nei_face]
            let nei_new_tet = nei_new_tets[nei_face];

            // Update my pointer to point to the neighbor's new tet at face 3
            set_opp_at(my_tet, my_face, encode_opp(nei_new_tet, 3u));

            // Also update the neighbor's backpointer to point to me at face 3
            set_opp_at(nei_new_tet, 3u, encode_opp(my_tet, my_face));
        }
    }
}
