// Port of kerUpdateOpp from gDel3D/GPU/KerDivision.cu (lines 561-695)
// Updates adjacency after flips (both 2-3 and 3-2)
//
// This is the CRITICAL kernel for flip correctness. It handles:
// 1. External adjacency extraction from old tets
// 2. Concurrent flip detection via tetMsgArr
// 3. Internal adjacency setup for new tets
//
// FlipItem structure (stored as 2 vec4<i32>):
//   _v[5]: vertices {v0, v1, v2, v3, v4}
//   _t[3]: tet indices {t0, t1, t2}
// Layout: [_v[0-3]], [_v[4], _t[0], _t[1], _t[2]]

@group(0) @binding(0) var<storage, read_write> flip_vec: array<vec4<i32>>; // FlipItem as 2 vec4s
@group(0) @binding(1) var<storage, read_write> tet_opp: array<u32>; // Flat array (4 u32s per tet)
@group(0) @binding(2) var<storage, read> tet_msg_arr: array<vec2<i32>>; // int2 messages
@group(0) @binding(3) var<storage, read> encoded_face_vi_arr: array<i32>;
@group(0) @binding(4) var<uniform> params: vec4<u32>; // x = org_flip_num, y = flip_num

// Flip32NewFaceVi lookup table from GPUDecl.h:149
const FLIP32_NEW_FACE_VI: array<array<u32, 2>, 3> = array<array<u32, 2>, 3>(
    array<u32, 2>(2u, 1u), // [0]
    array<u32, 2>(1u, 2u), // [1]
    array<u32, 2>(0u, 0u)  // [2]
);

// TetOpp encoding: (tet_idx << 5) | opp_vi
fn decode_opp(opp_val: u32) -> vec2<u32> {
    return vec2<u32>(opp_val >> 5u, opp_val & 31u);
}

fn encode_opp(tet_idx: u32, opp_vi: u32) -> u32 {
    return (tet_idx << 5u) | opp_vi;
}

// Internal adjacency encoding: (tet_idx << 5) | (1 << 2) | opp_vi (sets bit 2)
fn encode_opp_internal(tet_idx: u32, opp_vi: u32) -> u32 {
    return (tet_idx << 5u) | (1u << 2u) | opp_vi;
}

// Extract functions for tetMsg concurrent flip messaging
fn get_tet_idx(input: i32, old_vi: u32) -> u32 {
    let idx_vi = (input >> (old_vi * 4u)) & 0xF;
    return u32((idx_vi >> 2) & 0x3);
}

fn get_tet_vi(input: i32, old_vi: u32) -> u32 {
    let idx_vi = (input >> (old_vi * 4u)) & 0xF;
    return u32(idx_vi & 0x3);
}

// Load FlipItem (3 tet indices only, we don't need vertices for this kernel)
fn load_flip_tet_idx(flip_idx: u32) -> vec3<i32> {
    // FlipItem stored as 2 vec4s: [_v[0-3]], [_v[4], _t[0], _t[1], _t[2]]
    let t = flip_vec[flip_idx * 2u + 1u];
    return vec3<i32>(t.y, t.z, t.w); // _t[0], _t[1], _t[2]
}

@compute @workgroup_size(64)
fn update_opp(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let org_flip_num = i32(params.x);
    let flip_num = params.y;

    let flip_idx = gid.x;
    if flip_idx >= flip_num {
        return;
    }

    let flip_item = load_flip_tet_idx(flip_idx);

    // Determine flip type: Flip32 if _t[2] < 0, else Flip23
    let is_flip32 = flip_item.z < 0;

    let encoded_face_vi = encoded_face_vi_arr[flip_idx];

    // Extract external adjacency (6 slots for 2-3, but only some used for 3-2)
    var ext_opp: array<u32, 6>;

    // Load opp from first tet
    var opp_arr_0: array<u32, 4>;
    let tet_0 = u32(flip_item.x);
    for (var vi = 0u; vi < 4u; vi++) {
        opp_arr_0[vi] = tet_opp[tet_0 * 4u + vi];
    }

    if is_flip32 {
        // Flip32: 3->2 flip
        ext_opp[0] = opp_arr_0[(encoded_face_vi >> 10) & 3u];
        ext_opp[1] = opp_arr_0[(encoded_face_vi >> 8) & 3u];
    } else {
        // Flip23: 2->3 flip
        ext_opp[0] = opp_arr_0[(encoded_face_vi >> 10) & 3u];
        ext_opp[2] = opp_arr_0[(encoded_face_vi >> 6) & 3u];
        ext_opp[4] = opp_arr_0[(encoded_face_vi >> 2) & 3u];
    }

    // Load opp from second tet
    var opp_arr_1: array<u32, 4>;
    let tet_1 = u32(flip_item.y);
    for (var vi = 0u; vi < 4u; vi++) {
        opp_arr_1[vi] = tet_opp[tet_1 * 4u + vi];
    }

    if is_flip32 {
        // Flip32
        ext_opp[2] = opp_arr_1[(encoded_face_vi >> 6) & 3u];
        ext_opp[3] = opp_arr_1[(encoded_face_vi >> 4) & 3u];

        // Load opp from third tet (positive index)
        let tet_2 = u32(-flip_item.z); // makePositive
        var opp_arr_2: array<u32, 4>;
        for (var vi = 0u; vi < 4u; vi++) {
            opp_arr_2[vi] = tet_opp[tet_2 * 4u + vi];
        }

        ext_opp[4] = opp_arr_2[(encoded_face_vi >> 2) & 3u];
        ext_opp[5] = opp_arr_2[(encoded_face_vi >> 0) & 3u];
    } else {
        // Flip23
        ext_opp[1] = opp_arr_1[(encoded_face_vi >> 8) & 3u];
        ext_opp[3] = opp_arr_1[(encoded_face_vi >> 4) & 3u];
        ext_opp[5] = opp_arr_1[(encoded_face_vi >> 0) & 3u];
    }

    // Update with neighbors (handle concurrent flips)
    for (var i = 0u; i < 6u; i++) {
        let decoded = decode_opp(ext_opp[i]);
        var opp_idx = decoded.x;
        var opp_vi = decoded.y;

        // Check for invalid neighbor
        if ext_opp[i] == 0xFFFFFFFFu {
            continue;
        }

        let msg = tet_msg_arr[opp_idx];

        if msg.y < org_flip_num {
            // Neighbor not flipped - update neighbor's opp to point to new tet
            var new_tet_idx: i32;
            var vi: u32;

            if !is_flip32 {
                // Flip23
                new_tet_idx = flip_item[i / 2u];
                vi = select(0u, 3u, (i & 1u) != 0u);
            } else {
                // Flip32
                new_tet_idx = flip_item[1u - (i & 1u)];
                vi = FLIP32_NEW_FACE_VI[i / 2u][i & 1u];
            }

            // Write to neighbor's adjacency
            tet_opp[opp_idx * 4u + opp_vi] = encode_opp(u32(new_tet_idx), vi);
        } else {
            // Neighbor was also flipped - update my own opp
            let opp_flip_idx = u32(msg.y - org_flip_num);

            let new_loc_opp_idx = get_tet_idx(msg.x, opp_vi);

            if new_loc_opp_idx != 3u {
                let opp_flip_item = load_flip_tet_idx(opp_flip_idx);
                opp_idx = u32(opp_flip_item[new_loc_opp_idx]);
            }

            opp_vi = get_tet_vi(msg.x, opp_vi);
            ext_opp[i] = encode_opp(opp_idx, opp_vi);
        }
    }

    // Now output adjacency for new tets
    var opp_out: array<u32, 4>;

    if !is_flip32 {
        // Flip23: output 3 tets

        // First tet
        opp_out[0] = ext_opp[0];
        opp_out[1] = encode_opp_internal(u32(flip_item.y), 2u);
        opp_out[2] = encode_opp_internal(u32(flip_item.z), 1u);
        opp_out[3] = ext_opp[1];
        for (var vi = 0u; vi < 4u; vi++) {
            tet_opp[tet_0 * 4u + vi] = opp_out[vi];
        }

        // Second tet
        opp_out[0] = ext_opp[2];
        opp_out[1] = encode_opp_internal(u32(flip_item.z), 2u);
        opp_out[2] = encode_opp_internal(u32(flip_item.x), 1u);
        opp_out[3] = ext_opp[3];
        for (var vi = 0u; vi < 4u; vi++) {
            tet_opp[tet_1 * 4u + vi] = opp_out[vi];
        }

        // Third tet
        opp_out[0] = ext_opp[4];
        opp_out[1] = encode_opp_internal(u32(flip_item.x), 2u);
        opp_out[2] = encode_opp_internal(u32(flip_item.y), 1u);
        opp_out[3] = ext_opp[5];
        let tet_2 = u32(flip_item.z);
        for (var vi = 0u; vi < 4u; vi++) {
            tet_opp[tet_2 * 4u + vi] = opp_out[vi];
        }
    } else {
        // Flip32: output 2 tets

        // First tet
        opp_out[0] = ext_opp[5];
        opp_out[1] = ext_opp[1];
        opp_out[2] = ext_opp[3];
        opp_out[3] = encode_opp_internal(u32(flip_item.y), 3u);
        for (var vi = 0u; vi < 4u; vi++) {
            tet_opp[tet_0 * 4u + vi] = opp_out[vi];
        }

        // Second tet
        opp_out[0] = ext_opp[4];
        opp_out[1] = ext_opp[2];
        opp_out[2] = ext_opp[0];
        opp_out[3] = encode_opp_internal(u32(flip_item.x), 3u);
        for (var vi = 0u; vi < 4u; vi++) {
            tet_opp[tet_1 * 4u + vi] = opp_out[vi];
        }
    }
}
