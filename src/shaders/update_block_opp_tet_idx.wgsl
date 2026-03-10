// Kernel: Update tet indices in adjacency array during block reordering
//
// Port of kerUpdateBlockOppTetIdx from KerDivision.cu (line 1015).
//
// After compaction or reordering, tet indices change. This kernel updates all
// adjacency pointers (opp) to reflect the new tet indices.
//
// There are two regions of tet indices:
// 1. Block region (< oldInfBlockIdx): Tets organized in blocks by insertion index
//    - These are remapped using: orderArr[insIdx] * MEAN_VERTEX_DEGREE + locIdx
// 2. Infinity region (>= oldInfBlockIdx): Special tets (like infinity boundary)
//    - These are shifted: tetIdx - oldInfBlockIdx + newInfBlockIdx
//
// CUDA source:
// ```
// for ( int idx = getCurThreadIdx(); idx < oldTetNum; idx += getThreadNum() ) {
//     TetOpp opp = loadOpp( oppArr, idx );
//     for ( int i = 0; i < 4; ++i ) {
//         int tetIdx = opp.getOppTet( i );
//         if ( tetIdx < 0 ) continue;
//         if ( tetIdx < oldInfBlockIdx ) {
//             int insIdx = tetIdx / MeanVertDegree;
//             int locIdx = tetIdx % MeanVertDegree;
//             opp.setOppTet( i, orderArr[ insIdx ] * MeanVertDegree + locIdx );
//         } else
//             opp.setOppTet( i, tetIdx - oldInfBlockIdx + newInfBlockIdx );
//     }
//     storeOpp( oppArr, idx, opp );
// }
// ```
//
// Dispatch: ceil(old_tet_num / 64)

@group(0) @binding(0) var<storage, read_write> opp_arr: array<vec4<u32>>; // TetOpp adjacency (4 neighbors per tet)
@group(0) @binding(1) var<storage, read> order_arr: array<u32>; // insertion reordering map
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = old_tet_num, y = old_inf_block_idx, z = new_inf_block_idx

const MEAN_VERTEX_DEGREE: u32 = 8u;
const INVALID: u32 = 0xFFFFFFFFu;

// TetOpp encoding: Each u32 stores (tet_idx << 5) | flags_and_vi (original CUDA encoding)
// Bits 0-1: vi, Bits 2-4: flags, Bits 5-31: tet_idx
fn get_opp_tet(opp_val: u32) -> u32 {
    if opp_val == INVALID {
        return INVALID;
    }
    return opp_val >> 5u;
}

fn set_opp_tet(opp_val: u32, new_tet_idx: u32) -> u32 {
    if opp_val == INVALID {
        return INVALID;
    }
    // Preserve lower 5 bits (flags and vi), replace upper bits with new tet index
    return (new_tet_idx << 5u) | (opp_val & 0x1Fu);
}

@compute @workgroup_size(256)
fn update_block_opp_tet_idx(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let old_tet_num = params.x;
    let old_inf_block_idx = params.y;
    let new_inf_block_idx = params.z;

    let idx = gid.x;
    if idx >= old_tet_num {
        return;
    }

    // Load the 4 adjacency values for this tet
    var opp = opp_arr[idx];

    // Update each of the 4 neighbors
    for (var i = 0u; i < 4u; i = i + 1u) {
        var opp_val: u32;
        if i == 0u { opp_val = opp.x; }
        else if i == 1u { opp_val = opp.y; }
        else if i == 2u { opp_val = opp.z; }
        else { opp_val = opp.w; }

        let tet_idx = get_opp_tet(opp_val);

        // Skip invalid neighbors
        if tet_idx == INVALID {
            continue;
        }

        var new_tet_idx: u32;

        if tet_idx < old_inf_block_idx {
            // Block region - remap via order array
            let ins_idx = tet_idx / MEAN_VERTEX_DEGREE;
            let loc_idx = tet_idx % MEAN_VERTEX_DEGREE;
            new_tet_idx = order_arr[ins_idx] * MEAN_VERTEX_DEGREE + loc_idx;
        } else {
            // Infinity region - shift
            new_tet_idx = tet_idx - old_inf_block_idx + new_inf_block_idx;
        }

        // Update the adjacency value
        let new_opp_val = set_opp_tet(opp_val, new_tet_idx);

        if i == 0u { opp.x = new_opp_val; }
        else if i == 1u { opp.y = new_opp_val; }
        else if i == 2u { opp.z = new_opp_val; }
        else { opp.w = new_opp_val; }
    }

    // Store updated adjacency
    opp_arr[idx] = opp;
}
