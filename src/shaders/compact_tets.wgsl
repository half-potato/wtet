// kerCompactTets - Move alive tets to their new compacted positions
// Port from gDel3D/GDelFlipping/src/gDel3D/GPU/KerDivision.cu lines 1191-1233
//
// For each alive tet (idx >= newTetNum):
//   newTetIdx = prefixArr[idx]
//   Copy tet and opp to new position
//   Update adjacency:
//     - If neighbor >= newTetNum: remap using prefixArr
//     - If neighbor < newTetNum: update neighbor's opposite to point to newTetIdx
//
// This performs the actual compaction and fixes up adjacency.

@group(0) @binding(0) var<storage, read> tet_info: array<u32>;
@group(0) @binding(1) var<storage, read> prefix_arr: array<u32>;
@group(0) @binding(2) var<storage, read_write> tets: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> tet_opp: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: vec4<u32>; // x = new_tet_num, y = tet_info_num

const TET_ALIVE: u32 = 1u;
const INVALID: u32 = 0xFFFFFFFFu;

fn is_tet_alive(info: u32) -> bool {
    return (info & TET_ALIVE) != 0u;
}

fn decode_opp_tet(packed: u32) -> u32 {
    return packed >> 5u;
}

fn decode_opp_vi(packed: u32) -> u32 {
    return packed & 3u;
}

fn encode_opp(tet_idx: u32, opp_vi: u32) -> u32 {
    return (tet_idx << 5u) | opp_vi;
}

fn set_opp_tet(packed: u32, new_tet_idx: u32) -> u32 {
    // Keep lower 5 bits (vi + flags), replace upper bits with new tet index
    return (packed & 0x1Fu) | (new_tet_idx << 5u);
}

@compute @workgroup_size(64)
fn compact_tets(@builtin(global_invocation_id) gid: vec3<u32>) {
    let new_tet_num = params.x;
    let tet_info_num = params.y;
    let idx = new_tet_num + gid.x;

    if idx >= tet_info_num {
        return;
    }

    // Skip dead tets
    if !is_tet_alive(tet_info[idx]) {
        return;
    }

    // Get new index from compact map
    let new_tet_idx = prefix_arr[idx];

    // Copy tet to new position
    tets[new_tet_idx] = tets[idx];

    // Load and process adjacency
    var opp: array<u32, 4>;
    opp[0] = atomicLoad(&tet_opp[idx * 4u + 0u]);
    opp[1] = atomicLoad(&tet_opp[idx * 4u + 1u]);
    opp[2] = atomicLoad(&tet_opp[idx * 4u + 2u]);
    opp[3] = atomicLoad(&tet_opp[idx * 4u + 3u]);

    for (var vi = 0u; vi < 4u; vi++) {
        let opp_val = opp[vi];

        // Skip invalid adjacency (-1 as u32)
        if opp_val == INVALID {
            continue;
        }

        let opp_idx = decode_opp_tet(opp_val);

        if opp_idx >= new_tet_num {
            // Neighbor is in the compaction region - check if it's alive before remapping
            // CRITICAL: prefix_arr[opp_idx] only contains valid data for ALIVE tets!
            // For DEAD tets, prefix_arr[opp_idx] contains garbage (original prefix sum count)
            if is_tet_alive(tet_info[opp_idx]) {
                // Neighbor is alive and will be moved - use its new compacted index
                let opp_new_idx = prefix_arr[opp_idx];
                opp[vi] = set_opp_tet(opp_val, opp_new_idx);
            } else {
                // Neighbor is dead - adjacency is invalid
                opp[vi] = INVALID;
            }
        } else {
            // Neighbor is in stable region - should be alive, update its pointer to us
            if is_tet_alive(tet_info[opp_idx]) {
                let opp_vi = decode_opp_vi(opp_val);
                let old_val = atomicLoad(&tet_opp[opp_idx * 4u + opp_vi]);
                atomicStore(&tet_opp[opp_idx * 4u + opp_vi], set_opp_tet(old_val, new_tet_idx));
            } else {
                // Neighbor is dead - mark adjacency as invalid
                opp[vi] = INVALID;
            }
        }
    }

    // Store updated adjacency
    atomicStore(&tet_opp[new_tet_idx * 4u + 0u], opp[0]);
    atomicStore(&tet_opp[new_tet_idx * 4u + 1u], opp[1]);
    atomicStore(&tet_opp[new_tet_idx * 4u + 2u], opp[2]);
    atomicStore(&tet_opp[new_tet_idx * 4u + 3u], opp[3]);
}
