// Kernel: Update block vertex free list during compaction
//
// Port of kerUpdateBlockVertFreeList from KerDivision.cu (line 956).
//
// This kernel reorganizes the free lists for vertices after a compaction or reordering.
// When vertices are reordered via scatter_map, their free list blocks need to be updated
// to maintain the invariant that vertex V's free slots are at [V * MEAN_VERTEX_DEGREE, (V+1) * MEAN_VERTEX_DEGREE).
//
// For each slot in the free list:
// - If the vertex is new (scatter_map[vert] >= old_ins_num), initialize a fresh block
// - Otherwise, remap the slot index based on the old free list state
//
// CUDA source:
// ```
// int freeNum = insTetVec._num * MeanVertDegree;
// for ( int idx = getCurThreadIdx(); idx < freeNum; idx += getThreadNum() ) {
//     int insIdx  = idx / MeanVertDegree;
//     int locIdx  = idx % MeanVertDegree;
//     int vert    = insTetVec._arr[ insIdx ];
//     int freeIdx = vert * MeanVertDegree + locIdx;
//     int newIdx;
//     if ( scatterMap[ vert ] >= oldInsNum )     // New vert
//     {
//         newIdx = idx;
//         // Update free size for new vert
//         if ( locIdx == 0 )
//             vertFreeArr[ vert ] = MeanVertDegree;
//     }
//     else
//         newIdx = idx - locIdx + freeArr[ freeIdx ] % MeanVertDegree;
//     freeArr[ freeIdx ] = newIdx;
// }
// ```
//
// Dispatch: ceil((ins_tet_num * MEAN_VERTEX_DEGREE) / 64)

@group(0) @binding(0) var<storage, read> ins_tet_vec: array<u32>; // insTetVec._arr (vertex indices)
@group(0) @binding(1) var<storage, read_write> vert_free_arr: array<u32>; // per-vertex free counts
@group(0) @binding(2) var<storage, read_write> free_arr: array<u32>; // block-based free list
@group(0) @binding(3) var<storage, read> scatter_map: array<u32>; // vertex reordering map
@group(0) @binding(4) var<uniform> params: vec4<u32>; // x = ins_tet_num, y = old_ins_num

const MEAN_VERTEX_DEGREE: u32 = 8u;

@compute @workgroup_size(64)
fn update_block_vert_free_list(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let ins_tet_num = params.x;
    let old_ins_num = params.y;
    let free_num = ins_tet_num * MEAN_VERTEX_DEGREE;

    let idx = gid.x;
    if idx >= free_num {
        return;
    }

    let ins_idx = idx / MEAN_VERTEX_DEGREE;
    let loc_idx = idx % MEAN_VERTEX_DEGREE;
    let vert = ins_tet_vec[ins_idx];
    let free_idx = vert * MEAN_VERTEX_DEGREE + loc_idx;

    var new_idx: u32;

    if scatter_map[vert] >= old_ins_num {
        // New vertex - initialize with consecutive indices
        new_idx = idx;

        // Update free size for new vertex (only first thread per vertex)
        if loc_idx == 0u {
            vert_free_arr[vert] = MEAN_VERTEX_DEGREE;
        }
    } else {
        // Existing vertex - remap based on old free list
        new_idx = idx - loc_idx + (free_arr[free_idx] % MEAN_VERTEX_DEGREE);
    }

    free_arr[free_idx] = new_idx;
}
