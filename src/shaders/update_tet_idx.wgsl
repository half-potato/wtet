// Port of kerUpdateTetIdx from gDel3D/GPU/KerDivision.cu (lines 1050-1076)
// Remaps tet indices after block reordering
//
// CUDA code:
//   for ( int idx = getCurThreadIdx(); idx < idxVec._num; idx += getThreadNum() )
//   {
//       int tetIdx = idxVec._arr[ idx ];
//
//       int posTetIdx = ( tetIdx < 0 ? makePositive( tetIdx ) : tetIdx );
//
//       if ( posTetIdx < oldInfBlockIdx )
//       {
//           int insIdx = posTetIdx / MeanVertDegree;
//           int locIdx = posTetIdx % MeanVertDegree;
//
//           posTetIdx = orderArr[ insIdx ] * MeanVertDegree + locIdx;
//       } else
//           posTetIdx = posTetIdx - oldInfBlockIdx + newInfBlockIdx;
//
//       idxVec._arr[ idx ] = ( tetIdx < 0 ? makeNegative( posTetIdx ) : posTetIdx );
//   }

@group(0) @binding(0) var<storage, read_write> idx_vec: array<i32>;
@group(0) @binding(1) var<storage, read> order_arr: array<u32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = old_inf_block_idx, y = new_inf_block_idx, z = vec_size

const MEAN_VERTEX_DEGREE: u32 = 8u;

fn make_positive(val: i32) -> u32 {
    if val < 0 {
        return u32(-val);
    }
    return u32(val);
}

fn make_negative(val: u32) -> i32 {
    return -i32(val);
}

@compute @workgroup_size(64)
fn update_tet_idx(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let old_inf_block_idx = params.x;
    let new_inf_block_idx = params.y;
    let vec_size = params.z;

    let idx = gid.x;
    if idx >= vec_size {
        return;
    }

    let tet_idx = idx_vec[idx];
    var pos_tet_idx = make_positive(tet_idx);

    if pos_tet_idx < old_inf_block_idx {
        // Remap using order array
        let ins_idx = pos_tet_idx / MEAN_VERTEX_DEGREE;
        let loc_idx = pos_tet_idx % MEAN_VERTEX_DEGREE;

        pos_tet_idx = order_arr[ins_idx] * MEAN_VERTEX_DEGREE + loc_idx;
    } else {
        // Shift by delta
        pos_tet_idx = pos_tet_idx - old_inf_block_idx + new_inf_block_idx;
    }

    // Restore sign
    idx_vec[idx] = select(i32(pos_tet_idx), make_negative(pos_tet_idx), tet_idx < 0);
}
