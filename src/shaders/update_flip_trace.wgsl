// Kernel: Update flip trace for concurrent flipping
//
// This kernel maintains a linked list structure for tracking flips.
// Each flip item stores references to 3 tets that will be modified.
// The kernel updates tet_to_flip to create a chain from each tet to all flips
// that will modify it, allowing later detection of conflicting flips.
//
// Algorithm for each flip item:
// 1. For _t[0] and _t[1] (always valid tets):
//    - Read current tet_to_flip[tetIdx] (the "next flip" in chain)
//    - If -1, store (tetIdx << 1) | 0 in flip item (terminal marker)
//    - Otherwise store the current value (link to next flip)
//    - Update tet_to_flip[tetIdx] = (flipIdx << 1) | 1 (link to this flip)
//
// 2. For _t[2] (may be negative):
//    - If negative: it's encoded, decode with makePositive and only update tet_to_flip
//    - If positive: same logic as _t[0]/_t[1]
//
// The encoding scheme:
//   - (flipIdx << 1) | 1 = pointer to a flip item
//   - (tetIdx << 1) | 0 = terminal marker (no more flips for this tet)
//   - -1 = uninitialized
//
// Based on kerUpdateFlipTrace in KerDivision.cu:742

@group(0) @binding(0) var<storage, read_write> flip_arr: array<vec4<i32>>;
@group(0) @binding(1) var<storage, read_write> tet_to_flip: array<i32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = org_flip_num, y = flip_num

// Helper: Convert negative encoded value to positive index
// CUDA: makePositive(v) = -(v + 2)
fn make_positive(v: i32) -> i32 {
    return -(v + 2);
}

// Load FlipItem from flat buffer
// FlipItem has 8 i32s: _v[5], _t[3]
// Stored as two vec4<i32> at idx*2 and idx*2+1
struct FlipItem {
    v: array<i32, 5>,
    t: array<i32, 3>,
}

fn load_flip(idx: u32) -> FlipItem {
    let t1 = flip_arr[idx * 2u + 0u];
    let t2 = flip_arr[idx * 2u + 1u];

    var flip: FlipItem;
    flip.v[0] = t1.x;
    flip.v[1] = t1.y;
    flip.v[2] = t1.z;
    flip.v[3] = t1.w;
    flip.v[4] = t2.x;
    flip.t[0] = t2.y;
    flip.t[1] = t2.z;
    flip.t[2] = t2.w;

    return flip;
}

fn store_flip(idx: u32, flip: FlipItem) {
    let t1 = vec4<i32>(flip.v[0], flip.v[1], flip.v[2], flip.v[3]);
    let t2 = vec4<i32>(flip.v[4], flip.t[0], flip.t[1], flip.t[2]);

    flip_arr[idx * 2u + 0u] = t1;
    flip_arr[idx * 2u + 1u] = t2;
}

@compute @workgroup_size(256)
fn update_flip_trace(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let org_flip_num = params.x;
    let flip_num = params.y;

    if idx >= flip_num {
        return;
    }

    let flip_idx = i32(org_flip_num + idx);
    var flip_item = load_flip(org_flip_num + idx);

    // If all vertices are -1, this flip item is empty, skip it
    if flip_item.v[0] == -1 {
        return;
    }

    // Process _t[0]
    var tet_idx = flip_item.t[0];
    var next_flip = tet_to_flip[tet_idx];
    flip_item.t[0] = select(next_flip, (tet_idx << 1) | 0, next_flip == -1);
    tet_to_flip[tet_idx] = (flip_idx << 1) | 1;

    // Process _t[1]
    tet_idx = flip_item.t[1];
    next_flip = tet_to_flip[tet_idx];
    flip_item.t[1] = select(next_flip, (tet_idx << 1) | 0, next_flip == -1);
    tet_to_flip[tet_idx] = (flip_idx << 1) | 1;

    // Process _t[2] (may be negative)
    tet_idx = flip_item.t[2];

    if tet_idx < 0 {
        // Negative encoding: decode and only update tet_to_flip
        tet_idx = make_positive(tet_idx);
        tet_to_flip[tet_idx] = (flip_idx << 1) | 1;
    } else {
        // Positive: same as _t[0]/_t[1]
        next_flip = tet_to_flip[tet_idx];
        flip_item.t[2] = select(next_flip, (tet_idx << 1) | 0, next_flip == -1);
        tet_to_flip[tet_idx] = (flip_idx << 1) | 1;
    }

    store_flip(org_flip_num + idx, flip_item);
}
