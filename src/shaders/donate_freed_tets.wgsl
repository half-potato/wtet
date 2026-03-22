// donate_freed_tets.wgsl
// Post-flip pass: donate freed tets from 3-2 flips back to the per-vertex free list.
//
// WHY: In CUDA, slot allocation (kerAllocateFlip23Slot) runs as a SEPARATE kernel
// BEFORE kerFlip, so 3-2 donations can't be picked up by concurrent 2-3 allocations.
// Our combined flip kernel would race: 3-2 frees slot X → concurrent 2-3 allocates X
// → both flips reference same tet → flip trace chain corruption → wrong vert_tet.
//
// SOLUTION: 3-2 flips in flip.wgsl only mark tet_c as dead (no free list interaction).
// This shader runs AFTER the flip kernel to safely donate freed tets to the free list,
// making them available for the NEXT flip batch's 2-3 allocations.

@group(0) @binding(0) var<storage, read> flip_arr: array<vec4<i32>>; // FlipItem = 2 x vec4<i32>
@group(0) @binding(1) var<storage, read_write> free_arr: array<u32>;
@group(0) @binding(2) var<storage, read_write> vert_free_arr: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> block_owner: array<u32>;
@group(0) @binding(4) var<uniform> params: vec4<u32>; // x = org_flip_num, y = flip_num

const MEAN_VERTEX_DEGREE: u32 = 8u;

// CUDA: makePositive(v) = -(v + 2)
fn make_positive(v: i32) -> u32 {
    return u32(-(v + 2));
}

@compute @workgroup_size(256)
fn donate_freed_tets(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let org_flip_num = params.x;
    let flip_num = params.y;

    if idx >= flip_num {
        return;
    }

    let flip_idx = org_flip_num + idx;

    // Read second vec4 of FlipItem: (v4, t0, t1, t2)
    let fv1 = flip_arr[flip_idx * 2u + 1u];
    let t2 = fv1.w;

    // Only process 3-2 flips (t2 < 0 means Flip32)
    if t2 >= 0 {
        return;
    }

    // Decode freed tet: makePositive(t2) = -(t2 + 2)
    let tet_c = make_positive(t2);

    // Calculate block ownership (CUDA: KerDivision.cu:496-497)
    let blk_idx = tet_c / MEAN_VERTEX_DEGREE;
    let owner_vertex = block_owner[blk_idx];

    // Atomically allocate slot in owner's free list
    let free_slot = atomicAdd(&vert_free_arr[owner_vertex], 1u);

    // Store donated tet in owner's free list
    free_arr[owner_vertex * MEAN_VERTEX_DEGREE + free_slot] = tet_c;
}
