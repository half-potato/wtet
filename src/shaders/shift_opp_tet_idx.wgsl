// Port of kerShiftOppTetIdx from gDel3D/GPU/KerDivision.cu (lines 1078-1103)
// Shifts adjacency tet indices >= start by shift amount
//
// CUDA code:
//   for ( int idx = getCurThreadIdx(); idx < tetNum; idx += getThreadNum() )
//   {
//       TetOpp opp = loadOpp( oppArr, idx );
//
//       for ( int i = 0; i < 4; ++i )
//       {
//           if ( opp._t[ i ] < 0 ) continue;
//
//           const int oppIdx = opp.getOppTet( i );
//
//           if ( oppIdx >= start )
//               opp.setOppTet( i, oppIdx + shift );
//       }
//
//       storeOpp( oppArr, idx, opp );
//   }

@group(0) @binding(0) var<storage, read_write> tet_opp: array<u32>; // Flat array of TetOpp (4 u32s per tet)
@group(0) @binding(1) var<uniform> params: vec4<u32>; // x = tet_num, y = start, z = shift

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
fn shift_opp_tet_idx(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let tet_num = params.x;
    let start = params.y;
    let shift = params.z;

    let idx = gid.x;
    if idx >= tet_num {
        return;
    }

    // Load TetOpp for this tet
    var opp: array<u32, 4>;
    for (var vi = 0u; vi < 4u; vi++) {
        opp[vi] = tet_opp[idx * 4u + vi];
    }

    // Shift each adjacency if >= start
    for (var i = 0u; i < 4u; i++) {
        // Check for invalid adjacency (would be < 0 in CUDA)
        if opp[i] == 0xFFFFFFFFu {
            continue;
        }

        let decoded = decode_opp(opp[i]);
        var opp_idx = decoded.x;
        let opp_vi = decoded.y;

        if opp_idx >= start {
            opp_idx = opp_idx + shift;
            opp[i] = encode_opp(opp_idx, opp_vi);
        }
    }

    // Store updated adjacency
    for (var vi = 0u; vi < 4u; vi++) {
        tet_opp[idx * 4u + vi] = opp[vi];
    }
}
