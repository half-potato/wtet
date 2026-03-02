// Port of kerMakeReverseMap from gDel3D/GPU/KerDivision.cu (lines 816-832)
// Creates reverse mapping from scatter array
//
// CUDA code:
//   for ( int idx = getCurThreadIdx(); idx < insVertVec._num; idx += getThreadNum() )
//   {
//       const int oldIdx = scatterArr[ insVertVec._arr[ idx ] ];
//
//       if ( oldIdx < num )
//           revMapArr[ oldIdx ] = idx;
//   }

@group(0) @binding(0) var<storage, read> ins_vert_vec: array<u32>;
@group(0) @binding(1) var<storage, read> scatter_arr: array<u32>;
@group(0) @binding(2) var<storage, read_write> rev_map_arr: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x = ins_vert_num, y = num

@compute @workgroup_size(64)
fn make_reverse_map(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let ins_vert_num = params.x;
    let num = params.y;

    let idx = gid.x;
    if idx >= ins_vert_num {
        return;
    }

    let vert = ins_vert_vec[idx];
    let old_idx = scatter_arr[vert];

    if old_idx < num {
        rev_map_arr[old_idx] = idx;
    }
}
