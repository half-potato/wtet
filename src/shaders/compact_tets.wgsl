// Port of kerCompactTets from gDel3D/GPU/KerDivision.cu (lines 1191-1234)
// Moves alive tets to their new positions and updates adjacency
//
// CUDA code:
//   for ( int idx = newTetNum + getCurThreadIdx(); idx < tetInfoVec._num; idx += getThreadNum() )
//   {
//       if ( !isTetAlive( tetInfoVec._arr[ idx ] ) ) continue;
//       int newTetIdx   = prefixArr[ idx ];
//
//       Tet tet = loadTet( tetArr, idx );
//       storeTet( tetArr, newTetIdx, tet );
//
//       TetOpp opp = loadOpp( oppArr, idx );
//
//       for ( int vi = 0; vi < 4; ++vi )
//       {
//           if ( opp._t[ vi ] < 0 ) continue;
//           const int oppIdx = opp.getOppTet( vi );
//
//           if ( oppIdx >= newTetNum )
//           {
//               const int oppNewIdx = prefixArr[ oppIdx ];
//               opp.setOppTet( vi, oppNewIdx );
//           }
//           else
//           {
//               const int oppVi = opp.getOppVi( vi );
//               oppArr[ oppIdx ].setOppTet( oppVi, newTetIdx );
//           }
//       }
//
//       storeOpp( oppArr, newTetIdx, opp );
//   }

@group(0) @binding(0) var<storage, read> tet_info: array<u32>;
@group(0) @binding(1) var<storage, read> prefix_arr: array<u32>; // The compact map
@group(0) @binding(2) var<storage, read_write> tets: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> tet_opp: array<u32>; // Flat array of TetOpp (4 u32s per tet)
@group(0) @binding(4) var<uniform> params: vec4<u32>; // x = new_tet_num, y = total_tet_num

const TET_ALIVE: u32 = 1u;

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
fn compact_tets(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let new_tet_num = params.x;
    let total_tet_num = params.y;

    // Start from new_tet_num
    let idx = new_tet_num + gid.x;

    if idx >= total_tet_num {
        return;
    }

    // Skip dead tets
    if (tet_info[idx] & TET_ALIVE) == 0u {
        return;
    }

    // Get new index from compact map
    let new_tet_idx = prefix_arr[idx];

    // Move tet vertices
    let tet = tets[idx];
    tets[new_tet_idx] = tet;

    // Load and update adjacency (TetOpp)
    var opp: array<u32, 4>;
    for (var vi = 0u; vi < 4u; vi++) {
        opp[vi] = tet_opp[idx * 4u + vi];
    }

    // Update each adjacency link
    for (var vi = 0u; vi < 4u; vi++) {
        let opp_val = opp[vi];

        // Check for invalid adjacency (negative in CUDA, but we use max u32)
        if opp_val == 0xFFFFFFFFu {
            continue;
        }

        let decoded = decode_opp(opp_val);
        let opp_idx = decoded.x;
        let opp_vi_val = decoded.y;

        if opp_idx >= new_tet_num {
            // Neighbor is also being compacted, update our link to its new index
            let opp_new_idx = prefix_arr[opp_idx];
            opp[vi] = encode_opp(opp_new_idx, opp_vi_val);
        } else {
            // Neighbor is in the already-compacted region, update its link to us
            tet_opp[opp_idx * 4u + opp_vi_val] = encode_opp(new_tet_idx, vi);
        }
    }

    // Store updated adjacency at new position
    for (var vi = 0u; vi < 4u; vi++) {
        tet_opp[new_tet_idx * 4u + vi] = opp[vi];
    }
}
