// Port of kerAllocateFlip23Slot from gDel3D/GPU/KerDivision.cu (lines 888-953)
// Allocates tet slots for 2-3 flips which create 1 extra tet
//
// For each 2-3 flip, tries to allocate a slot from the block of one of the
// tet's vertices (for better cache locality). If no slots available, allocates
// from the infinity vertex block or requests expansion.

@group(0) @binding(0) var<storage, read> flip_to_tet: array<i32>;
@group(0) @binding(1) var<storage, read> tets: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> vert_free_arr: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read> free_arr: array<u32>;
@group(0) @binding(4) var<storage, read_write> flip23_new_slot: array<i32>;
@group(0) @binding(5) var<uniform> params: vec4<u32>; // x = flip_num, y = inf_idx, z = tet_num

const MEAN_VERTEX_DEGREE: u32 = 8u;

fn get_vote_tet_idx(vote_val: i32) -> u32 {
    return u32(vote_val) >> 8u;
}

fn get_vote_flip_info(vote_val: i32) -> u32 {
    return u32(vote_val) & 0xFFu;
}

fn get_flip_type(flip_info: u32) -> u32 {
    let bot_cor_ord_vi = (flip_info >> 2u) & 3u;
    if bot_cor_ord_vi == 3u {
        return 0u; // Flip23
    }
    return 1u; // Flip32
}

@compute @workgroup_size(256)
fn allocate_flip23_slot(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let flip_idx = gid.x;
    let flip_num = params.x;
    let inf_idx = params.y;
    let tet_num = params.z;

    if flip_idx >= flip_num {
        return;
    }

    let vote_val = flip_to_tet[flip_idx];
    let bot_tet_idx = get_vote_tet_idx(vote_val);
    let flip_info = get_vote_flip_info(vote_val);
    let f_type = get_flip_type(flip_info);

    if f_type != 0u {
        // Not a 2-3 flip, no allocation needed
        return;
    }

    // Load bottom tet
    let bot_tet = tets[bot_tet_idx];

    // Try to allocate from one of the vertices of the bottom tet
    var free_idx = -1;

    for (var vi = 0u; vi < 4u; vi++) {
        let vert = bot_tet[vi];

        if vert >= inf_idx {
            continue; // Skip infinity vertex
        }

        // Try to allocate from this vertex's block
        let old_val = atomicSub(&vert_free_arr[vert], 1);
        let loc_idx = old_val - 1;

        if loc_idx >= 0 {
            let array_idx = vert * MEAN_VERTEX_DEGREE + u32(loc_idx);
            free_idx = i32(free_arr[array_idx]);
            break;
        } else {
            // Failed to allocate - restore the counter we just decremented
            atomicAdd(&vert_free_arr[vert], 1);
        }
    }

    // Still no free slot? Try infinity vertex block
    if free_idx == -1 {
        let old_val = atomicSub(&vert_free_arr[inf_idx], 1);
        let loc_idx = old_val - 1;

        if loc_idx >= 0 {
            let array_idx = inf_idx * MEAN_VERTEX_DEGREE + u32(loc_idx);
            free_idx = i32(free_arr[array_idx]);
        } else {
            // Restore counter and allocate beyond current tet_num
            atomicAdd(&vert_free_arr[inf_idx], 1);
            free_idx = i32(tet_num) - loc_idx - 1;
        }
    }

    flip23_new_slot[flip_idx] = free_idx;
}
