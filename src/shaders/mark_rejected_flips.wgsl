// Kernel: Mark Rejected Flips
//
// Validates flip votes and marks rejected flips as -1.
// For each flip vote, ensures all participating tets agree on the flip.
// Optionally compacts valid flips into output array.
//
// Port of kerMarkRejectedFlips from KerDivision.cu:196

@group(0) @binding(0) var<storage, read_write> act_tet_vec: array<i32>;
@group(0) @binding(1) var<storage, read> tet_opp: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> tet_vote_arr: array<i32>;
@group(0) @binding(3) var<storage, read_write> tet_info: array<u32>;
@group(0) @binding(4) var<storage, read_write> vote_arr: array<i32>;
@group(0) @binding(5) var<storage, read_write> flip_to_tet: array<i32>;
@group(0) @binding(6) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params: vec4<u32>; // x = act_tet_num, y = vote_offset, z = compact_mode

const TET_CHECKED: u32 = 2u; // Bit 1 set
const INVALID: u32 = 0xFFFFFFFFu;

// TetViAsSeenFrom[vi] gives the 3 other vertices in counter-clockwise order when seen from vi
const TET_VI_AS_SEEN_FROM: array<vec3<u32>, 4> = array<vec3<u32>, 4>(
    vec3<u32>(1u, 3u, 2u), // seen from vertex 0
    vec3<u32>(0u, 2u, 3u), // seen from vertex 1
    vec3<u32>(0u, 3u, 1u), // seen from vertex 2
    vec3<u32>(0u, 1u, 2u), // seen from vertex 3
);

fn decode_opp_tet(opp: u32) -> u32 {
    return opp >> 5u;
}

fn decode_opp_vi(opp: u32) -> u32 {
    return opp & 3u;
}

fn get_vote_flip_info(vote_val: i32) -> u32 {
    return u32(vote_val) & 0x0Fu;
}

fn get_flip_bot_vi(flip_info: u32) -> u32 {
    return flip_info & 0x3u;
}

fn get_flip_bot_cor_ord_vi(flip_info: u32) -> u32 {
    return (flip_info >> 2u) & 0x3u;
}

fn is_flip23(flip_info: u32) -> bool {
    return get_flip_bot_cor_ord_vi(flip_info) == 3u;
}

// NOTE: Inlined set_tet_checked because WGSL doesn't allow passing storage pointers to functions

var<workgroup> flip_num: atomic<u32>;
var<workgroup> flip_offset: u32;

@compute @workgroup_size(64)
fn mark_rejected_flips(@builtin(global_invocation_id) gid: vec3<u32>,
                       @builtin(local_invocation_id) lid: vec3<u32>) {
    let act_tet_num = params.x;
    let vote_offset = i32(params.y);
    let compact_mode = params.z != 0u;

    // Reset workgroup counter
    if lid.x == 0u && compact_mode {
        atomicStore(&flip_num, 0u);
    }
    workgroupBarrier();

    let idx = gid.x;
    if idx >= act_tet_num {
        return;
    }

    var flip_val = -1;
    let tet_idx = act_tet_vec[idx];

    if tet_idx != -1 {
        flip_val = vote_arr[idx];

        if flip_val == -1 {
            // No flip from this tet
            tet_info[u32(tet_idx)] |= TET_CHECKED;
            act_tet_vec[idx] = -1;
        } else {
            // Check if all participating tets agree on this flip
            let vote_val = vote_offset + tet_idx;
            let bot_vote_val = tet_vote_arr[u32(tet_idx)];

            if bot_vote_val != vote_val {
                flip_val = -1;
            } else {
                let flip_info = get_vote_flip_info(flip_val);
                let bot_vi = get_flip_bot_vi(flip_info);

                // Check top tet
                let bot_opp = atomicLoad(&tet_opp[u32(tet_idx) * 4u + bot_vi]);

                // CRITICAL: Check for INVALID adjacency before decoding
                if bot_opp == INVALID {
                    flip_val = -1;
                } else {
                    let top_tet_idx = decode_opp_tet(bot_opp);
                    let top_vote_val = tet_vote_arr[top_tet_idx];

                    if top_vote_val != vote_val {
                        flip_val = -1;
                    } else if !is_flip23(flip_info) {
                        // For 3-2 flip, also check corner/side tet
                        let ord_vi = TET_VI_AS_SEEN_FROM[bot_vi];
                        let cor_ord_vi = get_flip_bot_cor_ord_vi(flip_info);
                        // CRITICAL FIX: Cannot index vec3 with variable - causes SIGSEGV
                        // Use select() instead: if cor_ord_vi==0->x, ==1->y, ==2->z
                        let bot_cor_vi = select(select(ord_vi.x, ord_vi.y, cor_ord_vi == 1u), ord_vi.z, cor_ord_vi == 2u);

                        // Side tet
                        let side_opp = atomicLoad(&tet_opp[u32(tet_idx) * 4u + bot_cor_vi]);

                        // CRITICAL: Check for INVALID adjacency before decoding
                        if side_opp == INVALID {
                            flip_val = -1;
                        } else {
                            let side_tet_idx = decode_opp_tet(side_opp);
                            let side_vote_val = tet_vote_arr[side_tet_idx];

                            if side_vote_val != vote_val {
                                flip_val = -1;
                            }
                        }
                    }
                }
            }

            if flip_val == -1 {
                vote_arr[idx] = -1;
            }
        }
    }

    // Compact valid flips if requested
    if compact_mode {
        var flip_loc_idx = -1;
        if flip_val != -1 {
            flip_loc_idx = i32(atomicAdd(&flip_num, 1u));
        }

        workgroupBarrier();

        let local_flip_num = atomicLoad(&flip_num);
        if local_flip_num > 0u {
            if lid.x == 0u {
                flip_offset = atomicAdd(&counters[0], local_flip_num);
            }
            workgroupBarrier();

            if flip_loc_idx != -1 {
                flip_to_tet[flip_offset + u32(flip_loc_idx)] = flip_val;
            }

            if lid.x == 0u {
                atomicStore(&flip_num, 0u);
            }
            workgroupBarrier();
        }
    }

    // Reset exact counter (only once)
    if gid.x == 0u {
        atomicStore(&counters[1], 0u);
    }
}
