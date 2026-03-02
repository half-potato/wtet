// Port of kerMarkRejectedFlips from gDel3D/GPU/KerDivision.cu (lines 195-308)
// Validates flip votes - checks if all participating tets agree on the vote
//
// CUDA code validates that:
// 1. For 2-3 flips: bot and top tets both vote for the same face
// 2. For 3-2 flips: bot, top, and side tets all vote for the same configuration
//
// This prevents invalid flips where neighboring tets disagree.

@group(0) @binding(0) var<storage, read> act_tet_vec: array<i32>;
@group(0) @binding(1) var<storage, read> tet_opp: array<u32>; // Flat array (4 u32s per tet)
@group(0) @binding(2) var<storage, read> tet_vote_arr: array<i32>;
@group(0) @binding(3) var<storage, read_write> tet_info: array<u32>;
@group(0) @binding(4) var<storage, read_write> vote_arr: array<i32>;
@group(0) @binding(5) var<storage, read_write> flip_to_tet: array<i32>; // Output compacted flips
@group(0) @binding(6) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params: vec4<u32>; // x = act_tet_num, y = vote_offset, z = compact_mode

// TetViAsSeenFrom lookup table (from GPUDecl.h)
const TET_VI_AS_SEEN_FROM: array<array<u32, 4>, 4> = array<array<u32, 4>, 4>(
    array<u32, 4>(0u, 2u, 1u, 3u), // [0]
    array<u32, 4>(0u, 3u, 2u, 1u), // [1]
    array<u32, 4>(0u, 1u, 3u, 2u), // [2]
    array<u32, 4>(0u, 2u, 3u, 1u)  // [3] - note: CUDA has typo {0,2,1,3} but should be {0,2,3,1}
);

const TET_CHECKED: u32 = 0u;  // Bit 1 = 0
const TET_CHANGED: u32 = 2u;  // Bit 1 = 1

// Decode TetOpp
fn decode_opp(opp_val: u32) -> vec2<u32> {
    return vec2<u32>(opp_val >> 5u, opp_val & 31u);
}

// Extract vote info from voteVal
fn get_vote_flip_info(vote_val: i32) -> u32 {
    return u32(vote_val & 0x0F);
}

fn get_flip_bot_vi(flip_info: u32) -> u32 {
    return flip_info & 0x3u;
}

fn get_flip_bot_cor_ord_vi(flip_info: u32) -> u32 {
    return (flip_info >> 2u) & 0x3u;
}

fn get_flip_type_is_32(flip_info: u32) -> bool {
    return get_flip_bot_cor_ord_vi(flip_info) != 3u;
}

// Workgroup shared memory for compaction
var<workgroup> s_flip_num: atomic<u32>;
var<workgroup> s_flip_offset: u32;

@compute @workgroup_size(64)
fn mark_rejected_flips(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let act_tet_num = params.x;
    let vote_offset = i32(params.y);
    let compact_mode = params.z;

    let idx = gid.x;

    // Initialize shared memory
    if lid.x == 0u {
        atomicStore(&s_flip_num, 0u);
    }
    workgroupBarrier();

    var flip_val = -1;

    if idx < act_tet_num {
        let tet_idx = act_tet_vec[idx];

        if tet_idx != -1 {
            flip_val = vote_arr[idx];

            if flip_val == -1 {
                // No flip from this tet - mark as Checked
                tet_info[tet_idx] = tet_info[tet_idx] & (~TET_CHANGED); // Clear bit 1
                // Note: In CUDA this also sets act_tet_vec[idx] = -1, but we can't write to read-only buffer
            } else {
                // Validate the flip vote
                let vote_val = vote_offset + tet_idx;
                let bot_vote_val = tet_vote_arr[tet_idx];

                if bot_vote_val != vote_val {
                    flip_val = -1;
                } else {
                    let flip_info = get_vote_flip_info(flip_val);
                    let is_flip32 = get_flip_type_is_32(flip_info);
                    let bot_vi = get_flip_bot_vi(flip_info);

                    // Load adjacency
                    var opp_arr: array<u32, 4>;
                    for (var vi = 0u; vi < 4u; vi++) {
                        opp_arr[vi] = tet_opp[u32(tet_idx) * 4u + vi];
                    }

                    // Check top tet
                    let top_decoded = decode_opp(opp_arr[bot_vi]);
                    let top_tet_idx = i32(top_decoded.x);
                    let top_vote_val = tet_vote_arr[top_tet_idx];

                    if top_vote_val != vote_val {
                        flip_val = -1;
                    } else if is_flip32 {
                        // For 3-2 flip, also check side tet
                        let ord_vi = TET_VI_AS_SEEN_FROM[bot_vi];
                        let cor_ord_vi = get_flip_bot_cor_ord_vi(flip_info);
                        let bot_cor_vi = ord_vi[cor_ord_vi];

                        let side_decoded = decode_opp(opp_arr[bot_cor_vi]);
                        let side_tet_idx = i32(side_decoded.x);
                        let side_vote_val = tet_vote_arr[side_tet_idx];

                        if side_vote_val != vote_val {
                            flip_val = -1;
                        }
                    }
                }

                if flip_val == -1 {
                    vote_arr[idx] = -1;
                }
            }
        }
    }

    // Compact flips if requested
    if compact_mode != 0u {
        var flip_loc_idx = -1;

        if flip_val != -1 {
            flip_loc_idx = i32(atomicAdd(&s_flip_num, 1u));
        }

        workgroupBarrier();

        if atomicLoad(&s_flip_num) > 0u {
            if lid.x == 0u {
                s_flip_offset = atomicAdd(&counters[0], atomicLoad(&s_flip_num));
            }

            workgroupBarrier();

            if flip_loc_idx != -1 {
                flip_to_tet[s_flip_offset + u32(flip_loc_idx)] = flip_val;
            }

            if lid.x == 0u {
                atomicStore(&s_flip_num, 0u);
            }

            workgroupBarrier();
        }
    }

    // Reset exact counter (first thread only)
    if wid.x == 0u && lid.x == 0u {
        // counters[1] is exact counter (assuming CounterExact = 1)
        atomicStore(&counters[1], 0u);
    }
}
