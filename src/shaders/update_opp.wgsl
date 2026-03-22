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

// Load tet indices from FlipItem (CUDA: loadFlipTetIdx)
// FlipItem layout: [_v[0], _v[1], _v[2], _v[3], _v[4], _t[0], _t[1], _t[2]]
// Second vec4 = (_v[4], _t[0], _t[1], _t[2]) → tet indices are .yzw
fn load_flip_tet_idx(flip_idx: u32) -> vec3<i32> {
    let item1 = flip_vec[flip_idx * 2u + 1u];
    return vec3<i32>(item1.y, item1.z, item1.w);
}

// CUDA convention: makePositive(v) = -(v + 2), inverse of makeNegative(v) = -(v + 2)
// For Flip32's _t[2] = -(tet_c + 2): make_positive(-(tet_c + 2)) = tet_c
fn make_positive(val: i32) -> u32 {
    return u32(-(val + 2));
}

// Strip stale OPP flags (OPP_INTERNAL=bit2, OPP_SPECIAL=bit3, OPP_SPHERE_FAIL=bit4)
// from external opp values inherited from old tets. Only preserves tet_idx and vi.
// CRITICAL: Must preserve INVALID (0xFFFFFFFF) values! Without this check,
// INVALID gets corrupted to 0xFFFFFFE3 which looks like a valid neighbor at
// out-of-bounds index, causing check_delaunay to test wrong vertex data.
fn clean_ext_opp(val: u32) -> u32 {
    if val == 0xFFFFFFFFu {
        return val;  // Preserve INVALID
    }
    return (val & 0xFFFFFFE0u) | (val & 3u);  // Keep bits 5+ (tet_idx) and bits 0-1 (vi)
}

@compute @workgroup_size(256)
fn update_opp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flip_idx = gid.x;
    let org_flip_num = params.x;
    let flip_num = params.y;

    if flip_idx >= flip_num {
        return;
    }

    // Load tet indices (CUDA: FlipItemTetIdx)
    // tet_idx.x = _t[0] = botTetIdx, .y = _t[1] = topTetIdx, .z = _t[2] = sideTetIdx
    // CRITICAL: Read at global index (org_flip_num + flip_idx) since flip.wgsl writes at global indices
    // CUDA: receives flipVec = _flipVec + orgFlipNum (offset pointer), so flipVec[flipIdx] = _flipVec[orgFlipNum + flipIdx]
    let glob_idx = org_flip_num + flip_idx;
    let tet_idx = load_flip_tet_idx(glob_idx);
    let is_flip23 = tet_idx.z >= 0;

    let encoded_face_vi = encoded_face_vi_arr[glob_idx];
    var ext_opp: array<u32, 6>;

    // Load external opposites from tet 0 (botTetIdx)
    // CRITICAL: Strip stale OPP_INTERNAL/SPECIAL/SPHERE_FAIL flags from old tet values.
    // These flags are only meaningful for the old tet's faces, not the new tet's external faces.
    // CUDA preserves stale flags because it re-checks ALL changed tets from both sides,
    // but our flip queue only contains new tets, so stale OPP_INTERNAL would cause
    // the flip shader to incorrectly skip legitimate external faces.
    let base0 = u32(tet_idx.x) * 4u;
    if is_flip23 {
        ext_opp[0] = clean_ext_opp(atomicLoad(&opp_arr[base0 + ((u32(encoded_face_vi) >> 10u) & 3u)]));
        ext_opp[2] = clean_ext_opp(atomicLoad(&opp_arr[base0 + ((u32(encoded_face_vi) >> 6u) & 3u)]));
        ext_opp[4] = clean_ext_opp(atomicLoad(&opp_arr[base0 + ((u32(encoded_face_vi) >> 2u) & 3u)]));
    } else {
        ext_opp[0] = clean_ext_opp(atomicLoad(&opp_arr[base0 + ((u32(encoded_face_vi) >> 10u) & 3u)]));
        ext_opp[1] = clean_ext_opp(atomicLoad(&opp_arr[base0 + ((u32(encoded_face_vi) >> 8u) & 3u)]));
    }

    // Load external opposites from tet 1 (topTetIdx)
    let base1 = u32(tet_idx.y) * 4u;
    if is_flip23 {
        ext_opp[1] = clean_ext_opp(atomicLoad(&opp_arr[base1 + ((u32(encoded_face_vi) >> 8u) & 3u)]));
        ext_opp[3] = clean_ext_opp(atomicLoad(&opp_arr[base1 + ((u32(encoded_face_vi) >> 4u) & 3u)]));
        ext_opp[5] = clean_ext_opp(atomicLoad(&opp_arr[base1 + ((u32(encoded_face_vi) >> 0u) & 3u)]));
    } else {
        ext_opp[2] = clean_ext_opp(atomicLoad(&opp_arr[base1 + ((u32(encoded_face_vi) >> 6u) & 3u)]));
        ext_opp[3] = clean_ext_opp(atomicLoad(&opp_arr[base1 + ((u32(encoded_face_vi) >> 4u) & 3u)]));

        // Load from tet 2 (sideTetIdx) for 3-2 flip
        let base2 = make_positive(tet_idx.z) * 4u;
        ext_opp[4] = clean_ext_opp(atomicLoad(&opp_arr[base2 + ((u32(encoded_face_vi) >> 2u) & 3u)]));
        ext_opp[5] = clean_ext_opp(atomicLoad(&opp_arr[base2 + ((u32(encoded_face_vi) >> 0u) & 3u)]));
    }

    // Update with neighbors (CUDA: KerDivision.cu:611-651)
    for (var i = 0u; i < 6u; i++) {
        var new_tet_idx: u32;
        var vi: u32;
        let tet_opp = ext_opp[i];

        var opp_idx = get_opp_val_tet(tet_opp);
        var opp_vi = get_opp_val_vi(tet_opp);

        let msg = tet_msg_arr[opp_idx];

        if msg.y < i32(org_flip_num) {
            // Neighbor not flipped - set my neighbor's opp
            if is_flip23 {
                // CUDA: newTetIdx = flipItem._t[i / 2]
                let idx = i / 2u;
                if idx == 0u { new_tet_idx = u32(tet_idx.x); }
                else if idx == 1u { new_tet_idx = u32(tet_idx.y); }
                else { new_tet_idx = u32(tet_idx.z); }
                vi = select(0u, 3u, (i & 1u) != 0u);
            } else {
                // CUDA: newTetIdx = flipItem._t[1 - (i & 1)]
                if (i & 1u) != 0u { new_tet_idx = u32(tet_idx.x); }
                else { new_tet_idx = u32(tet_idx.y); }
                // Flip32NewFaceVi[i/2][i&1] - explicit branches to avoid variable indexing SIGSEGV
                if i == 0u { vi = 2u; }       // [0][0] = 2
                else if i == 1u { vi = 1u; }  // [0][1] = 1
                else if i == 2u { vi = 1u; }  // [1][0] = 1
                else if i == 3u { vi = 2u; }  // [1][1] = 2
                else if i == 4u { vi = 0u; }  // [2][0] = 0
                else { vi = 0u; }             // [2][1] = 0
            }

            atomicStore(&opp_arr[opp_idx * 4u + opp_vi], make_opp_val(new_tet_idx, vi));
        } else {
            // Neighbor flipped - update my own opp
            let new_loc_opp_idx = get_tet_idx(msg.x, opp_vi);

            if new_loc_opp_idx != 3u {
                // CUDA: oppIdx = flipVec[oppFlipIdx]._t[newLocOppIdx]
                // msg.y is the global flip index, use directly since flip_arr uses global indices
                let opp_tet = load_flip_tet_idx(u32(msg.y));
                if new_loc_opp_idx == 0u { opp_idx = u32(opp_tet.x); }
                else if new_loc_opp_idx == 1u { opp_idx = u32(opp_tet.y); }
                else { opp_idx = u32(opp_tet.z); }
            }

            opp_vi = get_tet_vi(msg.x, opp_vi);
            ext_opp[i] = make_opp_val(opp_idx, opp_vi);
        }
    }

    // Write updated opp arrays (CUDA: KerDivision.cu:653-691)
    // Note: ext_opp values may carry stale flags (OPP_INTERNAL etc.) from the old
    // tets. CUDA preserves these too — stale flags are handled because checkDelaunay
    // retains rejected tets in the active list, so the face gets checked from the
    // neighbor's (clean) side.
    if is_flip23 {
        // Tet 0
        let out0 = u32(tet_idx.x) * 4u;
        atomicStore(&opp_arr[out0 + 0u], ext_opp[0]);
        atomicStore(&opp_arr[out0 + 1u], make_opp_val_internal(u32(tet_idx.y), 2u));
        atomicStore(&opp_arr[out0 + 2u], make_opp_val_internal(u32(tet_idx.z), 1u));
        atomicStore(&opp_arr[out0 + 3u], ext_opp[1]);

        // Tet 1
        let out1 = u32(tet_idx.y) * 4u;
        atomicStore(&opp_arr[out1 + 0u], ext_opp[2]);
        atomicStore(&opp_arr[out1 + 1u], make_opp_val_internal(u32(tet_idx.z), 2u));
        atomicStore(&opp_arr[out1 + 2u], make_opp_val_internal(u32(tet_idx.x), 1u));
        atomicStore(&opp_arr[out1 + 3u], ext_opp[3]);

        // Tet 2
        let out2 = u32(tet_idx.z) * 4u;
        atomicStore(&opp_arr[out2 + 0u], ext_opp[4]);
        atomicStore(&opp_arr[out2 + 1u], make_opp_val_internal(u32(tet_idx.x), 2u));
        atomicStore(&opp_arr[out2 + 2u], make_opp_val_internal(u32(tet_idx.y), 1u));
        atomicStore(&opp_arr[out2 + 3u], ext_opp[5]);
    } else {
        // 3-2 flip
        // Tet 0
        let out0 = u32(tet_idx.x) * 4u;
        atomicStore(&opp_arr[out0 + 0u], ext_opp[5]);
        atomicStore(&opp_arr[out0 + 1u], ext_opp[1]);
        atomicStore(&opp_arr[out0 + 2u], ext_opp[3]);
        atomicStore(&opp_arr[out0 + 3u], make_opp_val_internal(u32(tet_idx.y), 3u));

        // Tet 1
        let out1 = u32(tet_idx.y) * 4u;
        atomicStore(&opp_arr[out1 + 0u], ext_opp[4]);
        atomicStore(&opp_arr[out1 + 1u], ext_opp[2]);
        atomicStore(&opp_arr[out1 + 2u], ext_opp[0]);
        atomicStore(&opp_arr[out1 + 3u], make_opp_val_internal(u32(tet_idx.x), 3u));
    }
}
