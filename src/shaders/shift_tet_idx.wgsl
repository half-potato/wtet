// Port of kerShiftTetIdx from gDel3D/GPU/KerDivision.cu (lines 1105-1125)
// Shifts tet indices >= start by shift amount
//
// CUDA code:
//   int negStart = makeNegative( start );
//
//   for ( int idx = getCurThreadIdx(); idx < idxVec._num; idx += getThreadNum() )
//   {
//       const int oldIdx = idxVec._arr[ idx ];
//
//       if ( oldIdx >= start )
//           idxVec._arr[ idx ] = oldIdx + shift;
//       else if ( oldIdx < negStart )
//           idxVec._arr[ idx ] = oldIdx - shift;
//   }

@group(0) @binding(0) var<storage, read_write> idx_vec: array<i32>;
@group(0) @binding(1) var<uniform> params: vec4<u32>; // x = vec_size, y = start, z = shift

@compute @workgroup_size(64)
fn shift_tet_idx(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let vec_size = params.x;
    let start = i32(params.y);
    let shift = i32(params.z);

    let idx = gid.x;
    if idx >= vec_size {
        return;
    }

    let neg_start = -start;
    let old_idx = idx_vec[idx];

    if old_idx >= start {
        idx_vec[idx] = old_idx + shift;
    } else if old_idx < neg_start {
        idx_vec[idx] = old_idx - shift;
    }
}
