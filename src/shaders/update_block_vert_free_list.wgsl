// Port of kerUpdateBlockVertFreeList from gDel3D/GPU/KerDivision.cu (lines 956-990)
// Updates free list for vertex blocks after reordering
//
// CUDA code:
//   int freeNum = insTetVec._num * MeanVertDegree;
//
//   for ( int idx = getCurThreadIdx(); idx < freeNum; idx += getThreadNum() )
//   {
//       int insIdx  = idx / MeanVertDegree;
//       int locIdx  = idx % MeanVertDegree;
//       int vert    = insTetVec._arr[ insIdx ];
//       int freeIdx = vert * MeanVertDegree + locIdx;
//
//       int newIdx;
//
//       if ( scatterMap[ vert ] >= oldInsNum )     // New vert
//       {
//           newIdx = idx;
//
//           // Update free size for new vert
//           if ( locIdx == 0 )
//               vertFreeArr[ vert ] = MeanVertDegree;
//       }
//       else
//           newIdx = idx - locIdx + freeArr[ freeIdx ] % MeanVertDegree;
//
//       freeArr[ freeIdx ] = newIdx;
//   }

@group(0) @binding(0) var<storage, read> ins_tet_vec: array<u32>;
@group(0) @binding(1) var<storage, read_write> vert_free_arr: array<u32>;
@group(0) @binding(2) var<storage, read_write> free_arr: array<u32>;
@group(0) @binding(3) var<storage, read> scatter_map: array<u32>;
@group(0) @binding(4) var<uniform> params: vec4<u32>; // x = ins_num, y = old_ins_num

const MEAN_VERTEX_DEGREE: u32 = 8u;

@compute @workgroup_size(64)
fn update_block_vert_free_list(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let ins_num = params.x;
    let old_ins_num = params.y;
    let free_num = ins_num * MEAN_VERTEX_DEGREE;

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
        // New vertex
        new_idx = idx;

        // Update free size for new vert (only on first slot)
        if loc_idx == 0u {
            vert_free_arr[vert] = MEAN_VERTEX_DEGREE;
        }
    } else {
        // Existing vertex
        new_idx = idx - loc_idx + free_arr[free_idx] % MEAN_VERTEX_DEGREE;
    }

    free_arr[free_idx] = new_idx;
}
