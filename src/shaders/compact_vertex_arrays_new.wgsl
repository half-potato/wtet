// GPU compaction for vertex arrays (uninserted + vert_tet) - PREFIX SUM VERSION
// Removes vertices that were successfully inserted
// Replaces O(N²) linear search with O(N) mark + prefix sum + scatter
//
// Four-pass algorithm:
// Pass 1: Mark inserted positions (O(num_inserted))
// Pass 2: Invert to get non-inserted flags (O(num_uninserted))
// Pass 3: GPU prefix sum on flags (uses existing b0nes164 implementation)
// Pass 4: Scatter non-inserted vertices using compact map (O(num_uninserted))

@group(0) @binding(0) var<storage, read> uninserted_in: array<u32>;
@group(0) @binding(1) var<storage, read> vert_tet_in: array<u32>;
@group(0) @binding(2) var<storage, read> insert_list: array<vec2<u32>>; // (tet_idx, position)
@group(0) @binding(3) var<storage, read_write> uninserted_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> vert_tet_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> flags: array<u32>; // Workspace: is_not_inserted flags
@group(0) @binding(6) var<storage, read> compact_map: array<u32>; // Prefix sum result
@group(0) @binding(7) var<storage, read_write> counter: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> params: vec4<u32>;
// x = num_uninserted (input size)
// y = num_inserted (insert_list size)
// z = pass_num (0, 1, or 2)

// ┌────────────────────────────────────────────────────────────────────────┐
// │ CRITICAL: insert_list[i].y is POSITION (idx), NOT vertex ID!          │
// │ This matches CUDA kerPickWinnerPoint (KerDivision.cu:311-335)         │
// │ DO NOT CHANGE TO vert_idx - that breaks compaction logic!             │
// └────────────────────────────────────────────────────────────────────────┘

// Entry point 1: Mark inserted positions
@compute @workgroup_size(256)
fn mark_inserted(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_inserted = params.y;

    if idx >= num_inserted {
        return;
    }

    // Mark this position as inserted (buffer starts zero-initialized)
    let position = insert_list[idx].y;
    flags[position] = 1u; // 1 = inserted (will be excluded after inversion)
}

// Entry point 2: Invert flags to get non-inserted (also initializes flags for unmarked positions)
@compute @workgroup_size(256)
fn invert_flags(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_uninserted = params.x;

    if idx >= num_uninserted {
        return;
    }

    // Buffer starts zero-initialized:
    // - If mark_inserted set to 1: this position WAS inserted → invert to 0 (exclude)
    // - If still 0: this position was NOT inserted → invert to 1 (include)
    flags[idx] = 1u - flags[idx];
}

// Entry point 3: Scatter non-inserted vertices using compact map
@compute @workgroup_size(256)
fn scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_uninserted = params.x;

    if idx >= num_uninserted {
        return;
    }

    // If this vertex was NOT inserted, copy it to compacted output
    if flags[idx] == 1u {
        let out_idx = compact_map[idx]; // Exclusive prefix sum
        uninserted_out[out_idx] = uninserted_in[idx];
        vert_tet_out[out_idx] = vert_tet_in[idx];
    }
}
