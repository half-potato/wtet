// Kernel: Update Opp (Adjacency After Flips)
//
// Updates tet adjacency (opp) after performing flips.
// Handles both 2-3 and 3-2 flips, updating external and internal adjacencies.
//
// Port of kerUpdateOpp from KerDivision.cu:562

@group(0) @binding(0) var<storage, read> flip_vec: array<vec4<i32>>; // FlipItem = 2 x vec4<i32>
@group(0) @binding(1) var<storage, read_write> opp_arr: array<atomic<u32>>; // MUST be atomic to match other shaders
@group(0) @binding(2) var<storage, read> tet_msg_arr: array<vec2<i32>>;
@group(0) @binding(3) var<storage, read> encoded_face_vi_arr: array<i32>;
@group(0) @binding(4) var<uniform> params: vec4<u32>; // x = org_flip_num, y = flip_num

// Flip32NewFaceVi[3][2] = { {2,1}, {1,2}, {0,0} }
const FLIP32_NEW_FACE_VI: array<vec2<u32>, 3> = array<vec2<u32>, 3>(
    vec2<u32>(2u, 1u), // newTetIdx[0]'s vi, newTetIdx[1]'s vi
    vec2<u32>(1u, 2u),
    vec2<u32>(0u, 0u),
);

fn make_opp_val(tet_idx: u32, opp_tet_vi: u32) -> u32 {
    return (tet_idx << 5u) | opp_tet_vi;
}

fn make_opp_val_internal(tet_idx: u32, opp_tet_vi: u32) -> u32 {
    return (tet_idx << 5u) | (1u << 2u) | opp_tet_vi;
}

fn get_opp_val_tet(val: u32) -> u32 {
    return val >> 5u;
}

fn get_opp_val_vi(val: u32) -> u32 {
    return val & 3u;
}

fn get_tet_idx(input: i32, old_vi: u32) -> u32 {
    let idx_vi = (u32(input) >> (old_vi * 4u)) & 0xfu;
    return (idx_vi >> 2u) & 0x3u;
}

fn get_tet_vi(input: i32, old_vi: u32) -> u32 {
    let idx_vi = (u32(input) >> (old_vi * 4u)) & 0xfu;
    return idx_vi & 0x3u;
}

fn load_flip_item(flip_idx: u32) -> array<i32, 8> {
    let item0 = flip_vec[flip_idx * 2u];
    let item1 = flip_vec[flip_idx * 2u + 1u];
    return array<i32, 8>(
        item0.x, item0.y, item0.z, item0.w,
        item1.x, item1.y, item1.z, item1.w
    );
}

fn make_positive(val: i32) -> u32 {
    if val < 0 {
        return u32(-val);
    }
    return u32(val);
}

@compute @workgroup_size(64)
fn update_opp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flip_idx = gid.x;
    let org_flip_num = params.x;
    let flip_num = params.y;

    if flip_idx >= flip_num {
        return;
    }

    // Load flip item (8 ints total)
    let flip_item = load_flip_item(flip_idx);
    let is_flip23 = flip_item[2] >= 0;

    let encoded_face_vi = encoded_face_vi_arr[flip_idx];
    var ext_opp: array<u32, 6>;

    // Load external opposites from tet 0
    let base0 = u32(flip_item[0]) * 4u;
    if is_flip23 {
        ext_opp[0] = atomicLoad(&opp_arr[base0 + ((u32(encoded_face_vi) >> 10u) & 3u)]);
        ext_opp[2] = atomicLoad(&opp_arr[base0 + ((u32(encoded_face_vi) >> 6u) & 3u)]);
        ext_opp[4] = atomicLoad(&opp_arr[base0 + ((u32(encoded_face_vi) >> 2u) & 3u)]);
    } else {
        ext_opp[0] = atomicLoad(&opp_arr[base0 + ((u32(encoded_face_vi) >> 10u) & 3u)]);
        ext_opp[1] = atomicLoad(&opp_arr[base0 + ((u32(encoded_face_vi) >> 8u) & 3u)]);
    }

    // Load external opposites from tet 1
    let base1 = u32(flip_item[1]) * 4u;
    if is_flip23 {
        ext_opp[1] = atomicLoad(&opp_arr[base1 + ((u32(encoded_face_vi) >> 8u) & 3u)]);
        ext_opp[3] = atomicLoad(&opp_arr[base1 + ((u32(encoded_face_vi) >> 4u) & 3u)]);
        ext_opp[5] = atomicLoad(&opp_arr[base1 + ((u32(encoded_face_vi) >> 0u) & 3u)]);
    } else {
        ext_opp[2] = atomicLoad(&opp_arr[base1 + ((u32(encoded_face_vi) >> 6u) & 3u)]);
        ext_opp[3] = atomicLoad(&opp_arr[base1 + ((u32(encoded_face_vi) >> 4u) & 3u)]);

        // Load from tet 2 for 3-2 flip
        let base2 = make_positive(flip_item[2]) * 4u;
        ext_opp[4] = atomicLoad(&opp_arr[base2 + ((u32(encoded_face_vi) >> 2u) & 3u)]);
        ext_opp[5] = atomicLoad(&opp_arr[base2 + ((u32(encoded_face_vi) >> 0u) & 3u)]);
    }

    // // Update with neighbors
    for (var i = 0u; i < 6u; i++) {
        var new_tet_idx: u32;
        var vi: u32;
        let tet_opp = ext_opp[i];

        var opp_idx = get_opp_val_tet(tet_opp);
        var opp_vi = get_opp_val_vi(tet_opp);

        let msg = tet_msg_arr[opp_idx];

        if u32(msg.y) < org_flip_num {
            // Neighbor not flipped - set my neighbor's opp
            if is_flip23 {
                // flip_item[i / 2u] - avoid dynamic indexing
                let idx = i / 2u;
                new_tet_idx = select(select(u32(flip_item[0]), u32(flip_item[1]), idx == 1u), u32(flip_item[2]), idx == 2u);
                vi = select(0u, 3u, (i & 1u) != 0u);
            } else {
                // flip_item[1u - (i & 1u)] - avoid dynamic indexing
                new_tet_idx = select(u32(flip_item[1]), u32(flip_item[0]), (i & 1u) != 0u);
                vi = FLIP32_NEW_FACE_VI[i / 2u][i & 1u];
            }

            atomicStore(&opp_arr[opp_idx * 4u + opp_vi], make_opp_val(new_tet_idx, vi));
        } else {
            // Neighbor flipped - update my own opp
            let opp_flip_idx = u32(msg.y) - org_flip_num;
            let new_loc_opp_idx = get_tet_idx(msg.x, opp_vi);

            if new_loc_opp_idx != 3u {
                let opp_flip_item = load_flip_item(opp_flip_idx);
                opp_idx = u32(opp_flip_item[new_loc_opp_idx]);
            }

            opp_vi = get_tet_vi(msg.x, opp_vi);
            ext_opp[i] = make_opp_val(opp_idx, opp_vi);
        }
    }

    // Write updated opp arrays
    if is_flip23 {
        // Tet 0
        let out0 = u32(flip_item[0]) * 4u;
        atomicStore(&opp_arr[out0 + 0u], ext_opp[0]);
        atomicStore(&opp_arr[out0 + 1u], make_opp_val_internal(u32(flip_item[1]), 2u));
        atomicStore(&opp_arr[out0 + 2u], make_opp_val_internal(u32(flip_item[2]), 1u));
        atomicStore(&opp_arr[out0 + 3u], ext_opp[1]);

        // Tet 1
        let out1 = u32(flip_item[1]) * 4u;
        atomicStore(&opp_arr[out1 + 0u], ext_opp[2]);
        atomicStore(&opp_arr[out1 + 1u], make_opp_val_internal(u32(flip_item[2]), 2u));
        atomicStore(&opp_arr[out1 + 2u], make_opp_val_internal(u32(flip_item[0]), 1u));
        atomicStore(&opp_arr[out1 + 3u], ext_opp[3]);

        // Tet 2
        let out2 = u32(flip_item[2]) * 4u;
        atomicStore(&opp_arr[out2 + 0u], ext_opp[4]);
        atomicStore(&opp_arr[out2 + 1u], make_opp_val_internal(u32(flip_item[0]), 2u));
        atomicStore(&opp_arr[out2 + 2u], make_opp_val_internal(u32(flip_item[1]), 1u));
        atomicStore(&opp_arr[out2 + 3u], ext_opp[5]);
    } else {
        // 3-2 flip
        // Tet 0
        let out0 = u32(flip_item[0]) * 4u;
        atomicStore(&opp_arr[out0 + 0u], ext_opp[5]);
        atomicStore(&opp_arr[out0 + 1u], ext_opp[1]);
        atomicStore(&opp_arr[out0 + 2u], ext_opp[3]);
        atomicStore(&opp_arr[out0 + 3u], make_opp_val_internal(u32(flip_item[1]), 3u));

        // Tet 1
        let out1 = u32(flip_item[1]) * 4u;
        atomicStore(&opp_arr[out1 + 0u], ext_opp[4]);
        atomicStore(&opp_arr[out1 + 1u], ext_opp[2]);
        atomicStore(&opp_arr[out1 + 2u], ext_opp[0]);
        atomicStore(&opp_arr[out1 + 3u], make_opp_val_internal(u32(flip_item[0]), 3u));
    }
}
