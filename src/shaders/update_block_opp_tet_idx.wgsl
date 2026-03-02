// Port of kerUpdateBlockOppTetIdx from gDel3D/GPU/KerDivision.cu (lines 1015-1048)
// Remaps adjacency tet indices after block reordering
//
// CUDA code:
//   for ( int idx = getCurThreadIdx(); idx < oldTetNum; idx += getThreadNum() )
//   {
//       TetOpp opp = loadOpp( oppArr, idx );
//
//       for ( int i = 0; i < 4; ++i )
//       {
//           int tetIdx = opp.getOppTet( i );
//
//           if ( tetIdx < 0 ) continue;
//
//           if ( tetIdx < oldInfBlockIdx )
//           {
//               int insIdx = tetIdx / MeanVertDegree;
//               int locIdx = tetIdx % MeanVertDegree;
//
//               opp.setOppTet( i, orderArr[ insIdx ] * MeanVertDegree + locIdx );
//           }
//           else
//               opp.setOppTet( i, tetIdx - oldInfBlockIdx + newInfBlockIdx );
//       }
//
//       storeOpp( oppArr, idx, opp );
//   }

@group(0) @binding(0) var<storage, read_write> tet_opp: array<u32>; // Flat array of TetOpp (4 u32s per tet)
@group(0) @binding(1) var<storage, read> order_arr: array<u32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = old_inf_block_idx, y = new_inf_block_idx, z = old_tet_num

const MEAN_VERTEX_DEGREE: u32 = 8u;

// Helper to decode TetOpp entry: (tet_idx << 5) | opp_vi
fn decode_opp(opp_val: u32) -> vec2<u32> {
    let tet_idx = opp_val >> 5u;
    let opp_vi = opp_val & 31u;
    return vec2<u32>(tet_idx, opp_vi);
}

// Helper to encode TetOpp entry
fn encode_opp(tet_idx: u32, opp_vi: u32) -> u32 {
    return (tet_idx << 5u) | opp_vi;
}

@compute @workgroup_size(64)
fn update_block_opp_tet_idx(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let old_inf_block_idx = params.x;
    let new_inf_block_idx = params.y;
    let old_tet_num = params.z;

    let idx = gid.x;
    if idx >= old_tet_num {
        return;
    }

    // Load TetOpp for this tet
    var opp: array<u32, 4>;
    for (var vi = 0u; vi < 4u; vi++) {
        opp[vi] = tet_opp[idx * 4u + vi];
    }

    // Remap each adjacency
    for (var i = 0u; i < 4u; i++) {
        let decoded = decode_opp(opp[i]);
        var tet_idx = decoded.x;
        let opp_vi = decoded.y;

        // Check for invalid adjacency (would be negative in CUDA)
        if tet_idx == 0xFFFFFFFFu {
            continue;
        }

        if tet_idx < old_inf_block_idx {
            // Remap using order array
            let ins_idx = tet_idx / MEAN_VERTEX_DEGREE;
            let loc_idx = tet_idx % MEAN_VERTEX_DEGREE;

            tet_idx = order_arr[ins_idx] * MEAN_VERTEX_DEGREE + loc_idx;
        } else {
            // Shift by delta
            tet_idx = tet_idx - old_inf_block_idx + new_inf_block_idx;
        }

        opp[i] = encode_opp(tet_idx, opp_vi);
    }

    // Store updated adjacency
    for (var vi = 0u; vi < 4u; vi++) {
        tet_opp[idx * 4u + vi] = opp[vi];
    }
}
