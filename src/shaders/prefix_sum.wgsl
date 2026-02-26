// Work-efficient Blelloch parallel prefix sum (exclusive scan).
//
// Usage: Dispatch with ceil(n / (WORKGROUP_SIZE * 2)) workgroups.
// For arrays larger than one workgroup, a two-pass approach is needed:
//   1. Per-block scan + write block sums
//   2. Scan block sums
//   3. Add block sums back to each element

const WORKGROUP_SIZE: u32 = 256u;
const BLOCK_SIZE: u32 = 512u; // 2 * WORKGROUP_SIZE

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = n

var<workgroup> temp: array<u32, 512>;

// Pass 1: Per-block scan. Each workgroup handles BLOCK_SIZE elements.
// Writes the block's total to block_sums[workgroup_id.x].
@compute @workgroup_size(256)
fn scan_blocks(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let n = params.x;
    let block_offset = wid.x * BLOCK_SIZE;
    let tid = lid.x;

    // Load into shared memory
    let i0 = block_offset + tid;
    let i1 = block_offset + tid + WORKGROUP_SIZE;
    temp[tid] = select(0u, data[i0], i0 < n);
    temp[tid + WORKGROUP_SIZE] = select(0u, data[i1], i1 < n);

    // Up-sweep (reduce)
    var offset = 1u;
    for (var d = BLOCK_SIZE >> 1u; d > 0u; d >>= 1u) {
        workgroupBarrier();
        if tid < d {
            let ai = offset * (2u * tid + 1u) - 1u;
            let bi = offset * (2u * tid + 2u) - 1u;
            temp[bi] += temp[ai];
        }
        offset <<= 1u;
    }

    // Save block sum and clear last element
    workgroupBarrier();
    if tid == 0u {
        block_sums[wid.x] = temp[BLOCK_SIZE - 1u];
        temp[BLOCK_SIZE - 1u] = 0u;
    }

    // Down-sweep
    for (var d = 1u; d < BLOCK_SIZE; d <<= 1u) {
        offset >>= 1u;
        workgroupBarrier();
        if tid < d {
            let ai = offset * (2u * tid + 1u) - 1u;
            let bi = offset * (2u * tid + 2u) - 1u;
            let t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    // Write results back
    workgroupBarrier();
    if i0 < n { data[i0] = temp[tid]; }
    if i1 < n { data[i1] = temp[tid + WORKGROUP_SIZE]; }
}

// Pass 2: Add block sums back to each element.
@compute @workgroup_size(256)
fn add_block_sums(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let n = params.x;
    let block_offset = wid.x * BLOCK_SIZE;
    let block_sum = block_sums[wid.x];

    let i0 = block_offset + lid.x;
    let i1 = block_offset + lid.x + WORKGROUP_SIZE;
    if i0 < n { data[i0] += block_sum; }
    if i1 < n { data[i1] += block_sum; }
}
