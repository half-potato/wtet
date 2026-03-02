// Port of kerCollectFreeSlots from gDel3D/GPU/KerDivision.cu (lines 1152-1169)
// Collects indices of dead tets into the free array for reuse
//
// CUDA code:
//   for ( int idx = getCurThreadIdx(); idx < newTetNum; idx += getThreadNum() )
//   {
//       if ( isTetAlive( tetInfoArr[ idx ] ) ) continue;
//       int freeIdx = idx - prefixArr[ idx ];
//       freeArr[ freeIdx ] = idx;
//   }

@group(0) @binding(0) var<storage, read> tet_info: array<u32>;
@group(0) @binding(1) var<storage, read> prefix_sum: array<u32>; // Prefix sum of alive flags
@group(0) @binding(2) var<storage, read_write> free_slots: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x = num_tets

const TET_ALIVE: u32 = 1u;

@compute @workgroup_size(64)
fn collect_free_slots(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_tets = params.x;

    if idx >= num_tets {
        return;
    }

    // Skip alive tets
    if (tet_info[idx] & TET_ALIVE) != 0u {
        return;
    }

    // Dead tet - add to free list
    // freeIdx = idx - prefixArr[idx] (number of alive tets before this one)
    let free_idx = idx - prefix_sum[idx];
    free_slots[free_idx] = idx;
}
