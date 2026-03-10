// Kernels for compacting the tet array (removing dead tets).
//
// Two-pass approach:
//   1. kerMakeCompactMap: Write 1 for alive tets, 0 for dead → prefix sum → compact map
//   2. kerCompactTets: Use compact map to move tets to contiguous positions

@group(0) @binding(0) var<storage, read> tet_info: array<u32>;
@group(0) @binding(1) var<storage, read_write> alive_flags: array<u32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = max_tets

const TET_ALIVE: u32 = 1u;

@compute @workgroup_size(256)
fn make_compact_flags(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    if gid.x >= params.x {
        return;
    }
    alive_flags[gid.x] = select(0u, 1u, (tet_info[gid.x] & TET_ALIVE) != 0u);
}

// After prefix sum on alive_flags, compact_map[i] = new index for tet i.

@group(0) @binding(0) var<storage, read> src_tets: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> src_opp: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> src_info: array<u32>;
@group(0) @binding(3) var<storage, read> compact_map: array<u32>; // exclusive prefix sum of alive flags
@group(0) @binding(4) var<storage, read> alive: array<u32>;        // original alive flags
@group(0) @binding(5) var<storage, read_write> dst_tets: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read_write> dst_opp: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read_write> dst_info: array<u32>;
@group(0) @binding(8) var<uniform> params2: vec4<u32>; // x = max_tets

const INVALID: u32 = 0xFFFFFFFFu;

fn remap_opp(packed: u32, cmap: ptr<storage, array<u32>, read>) -> u32 {
    if packed == INVALID {
        return INVALID;
    }
    let old_tet = packed >> 5u;
    let face = packed & 3u;
    let new_tet = (*cmap)[old_tet];
    return (new_tet << 5u) | face;
}

@compute @workgroup_size(256)
fn compact_tets(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let old_idx = gid.x;
    if old_idx >= params2.x {
        return;
    }
    if alive[old_idx] == 0u {
        return;
    }

    let new_idx = compact_map[old_idx];

    dst_tets[new_idx] = src_tets[old_idx];
    dst_info[new_idx] = src_info[old_idx];

    // Remap adjacency pointers
    let old_opp = src_opp[old_idx];
    dst_opp[new_idx] = vec4<u32>(
        remap_opp(old_opp.x, &compact_map),
        remap_opp(old_opp.y, &compact_map),
        remap_opp(old_opp.z, &compact_map),
        remap_opp(old_opp.w, &compact_map),
    );
}
