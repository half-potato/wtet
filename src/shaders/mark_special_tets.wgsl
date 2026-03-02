// Port of kerMarkSpecialTets from gDel3D/GPU/KerDivision.cu (lines 834-870)
// Clears special adjacency markers and marks tets as Changed
//
// CUDA code:
//   for ( int idx = getCurThreadIdx(); idx < tetInfoVec._num; idx += getThreadNum() )
//   {
//       if ( !isTetAlive( tetInfoVec._arr[ idx ] ) ) continue;
//       TetOpp opp = loadOpp( oppArr, idx );
//       bool changed = false;
//
//       for ( int vi = 0; vi < 4; ++vi )
//       {
//           if ( opp.isOppSpecial( vi ) )
//           {
//               changed = true;
//               opp.setOppSpecial( vi, false );
//           }
//       }
//
//       if ( changed )
//       {
//           setTetCheckState( tetInfoVec._arr[ idx ], Changed );
//           storeOpp( oppArr, idx, opp );
//       }
//   }

@group(0) @binding(0) var<storage, read_write> tet_info: array<u32>;
@group(0) @binding(1) var<storage, read_write> tet_opp: array<u32>; // Flat array of TetOpp (4 u32s per tet)
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = num_tets

const TET_ALIVE: u32 = 1u;       // Bit 0
const TET_CHANGED: u32 = 2u;     // Bit 1
const OPP_SPECIAL: u32 = 8u;     // Bit 3 in TetOpp value

@compute @workgroup_size(64)
fn mark_special_tets(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_tets = params.x;

    if idx >= num_tets {
        return;
    }

    // Skip dead tets
    if (tet_info[idx] & TET_ALIVE) == 0u {
        return;
    }

    // Load TetOpp for this tet
    var opp: array<u32, 4>;
    for (var vi = 0u; vi < 4u; vi++) {
        opp[vi] = tet_opp[idx * 4u + vi];
    }

    var changed = false;

    // Check each adjacency for special marker
    for (var vi = 0u; vi < 4u; vi++) {
        if (opp[vi] & OPP_SPECIAL) != 0u {
            changed = true;
            // Clear special bit (bit 3)
            opp[vi] = opp[vi] & (~OPP_SPECIAL);
        }
    }

    // If any special markers were found, mark tet as Changed and store updated opp
    if changed {
        // Set Changed state (bit 1)
        tet_info[idx] = tet_info[idx] | TET_CHANGED;

        // Store updated adjacency
        for (var vi = 0u; vi < 4u; vi++) {
            tet_opp[idx * 4u + vi] = opp[vi];
        }
    }
}
