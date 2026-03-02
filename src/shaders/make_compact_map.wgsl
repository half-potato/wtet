// Port of kerMakeCompactMap from gDel3D/GPU/KerDivision.cu (lines 1171-1189)
// For alive tets beyond newTetNum, maps old index -> new index using free slots
//
// CUDA code:
//   for ( int idx = newTetNum + getCurThreadIdx(); idx < tetInfoVec._num; idx += getThreadNum() )
//   {
//       if ( !isTetAlive( tetInfoVec._arr[ idx ] ) ) continue;
//       int freeIdx     = newTetNum - prefixArr[ idx ];
//       int newTetIdx   = freeArr[ freeIdx ];
//       prefixArr[ idx ] = newTetIdx;
//   }

@group(0) @binding(0) var<storage, read> tet_info: array<u32>;
@group(0) @binding(1) var<storage, read_write> prefix_arr: array<u32>; // prefix sum, then becomes the compact map
@group(0) @binding(2) var<storage, read> free_arr: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x = new_tet_num, y = total_tet_num

const TET_ALIVE: u32 = 1u;

@compute @workgroup_size(64)
fn make_compact_map(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let new_tet_num = params.x;
    let total_tet_num = params.y;

    // Start from new_tet_num and process every thread-stride
    let idx = new_tet_num + gid.x;

    if idx >= total_tet_num {
        return;
    }

    // Skip dead tets
    if (tet_info[idx] & TET_ALIVE) == 0u {
        return;
    }

    // Map this alive tet to a free slot
    let free_idx = new_tet_num - prefix_arr[idx];
    let new_tet_idx = free_arr[free_idx];

    // Store the mapping
    prefix_arr[idx] = new_tet_idx;
}
