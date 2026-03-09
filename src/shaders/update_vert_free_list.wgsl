// update_vert_free_list.wgsl
// Port of CUDA kerUpdateVertFreeList (KerDivision.cu:1126-1149)
//
// PURPOSE: Reinitialize free lists for newly inserted vertices
// This is CRITICAL for masking the allocation/donation index mismatch bug
// documented in CUDA_FLAWS.md
//
// CUDA calls this EVERY iteration during expandTetraList()
// Without it, positions [7,6,5,4] in free_arr become stale and get reallocated
//
// CUDA:
//   int newFreeNum = insTetVec._num * MeanVertDegree;
//   for (int idx = ...; idx < newFreeNum; idx++)
//   {
//       int insIdx  = idx / MeanVertDegree;
//       int locIdx  = idx % MeanVertDegree;
//       int vertIdx = insTetVec._arr[insIdx];
//       freeArr[vertIdx * MeanVertDegree + locIdx] = startFreeIdx + idx;
//       if (idx < insTetVec._num)
//           vertFreeArr[insTetVec._arr[idx]] = MeanVertDegree;
//   }

@group(0) @binding(0) var<storage, read> insert_list: array<vec2<u32>>;  // [(tet_idx, position), ...]
@group(0) @binding(1) var<storage, read_write> vert_free_arr: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> free_arr: array<u32>;
@group(0) @binding(3) var<storage, read> uninserted: array<u32>;  // Map position → vertex_id
@group(0) @binding(4) var<uniform> params: Params;

struct Params {
    num_inserted: u32,      // insTetVec._num (number of vertices being inserted)
    start_free_idx: u32,    // startFreeIdx (base tet index for new allocations)
    _pad0: u32,
    _pad1: u32,
}

const MEAN_VERTEX_DEGREE: u32 = 8u;

@compute @workgroup_size(256)
fn update_vert_free_list(@builtin(global_invocation_id) gid: vec3<u32>) {
    let new_free_num = params.num_inserted * MEAN_VERTEX_DEGREE;
    let idx = gid.x;

    if idx >= new_free_num {
        return;
    }

    // CUDA line 1139-1141
    let ins_idx = idx / MEAN_VERTEX_DEGREE;  // Which vertex (0..num_inserted-1)
    let loc_idx = idx % MEAN_VERTEX_DEGREE;  // Which slot within vertex's block (0..7)

    // Extract vertex ID from insert_list: [tet_idx, position] → uninserted[position] → vertex_id
    let position = insert_list[ins_idx].y;
    let vert_idx = uninserted[position];

    // CUDA line 1143: Reinitialize this slot with sequential tet index
    // This OVERWRITES any stale values from previous allocations
    free_arr[vert_idx * MEAN_VERTEX_DEGREE + loc_idx] = params.start_free_idx + idx;

    // CUDA line 1146-1147: Reset free count to 8 (once per vertex)
    if idx < params.num_inserted {
        let pos = insert_list[idx].y;
        let v = uninserted[pos];
        atomicStore(&vert_free_arr[v], MEAN_VERTEX_DEGREE);
    }
}
