// Kernels: Vote for which point to insert into each tetrahedron.
//
// Two-pass approach:
//   1. kerVoteForPoint: Each uninserted point votes for its containing tet.
//      Uses atomicMin to pick the closest point (by distance to circumcenter or centroid).
//   2. kerPickWinnerPoint: Each tet that received a vote marks the winner.

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> tets: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> tet_info: array<u32>;
@group(0) @binding(3) var<storage, read_write> tet_vote: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read> vert_tet: array<u32>;
@group(0) @binding(5) var<storage, read> uninserted: array<u32>;
@group(0) @binding(6) var<uniform> params: vec4<u32>; // x = num_uninserted

const INVALID: u32 = 0xFFFFFFFFu;
const NO_VOTE: i32 = 0x7FFFFFFF;
const TET_ALIVE: u32 = 1u;

// Pack distance (float, discretized) and vertex index into i32 for atomicMin.
// High 16 bits: distance key, low 16 bits: vertex index within chunk.
// This limits us to 65536 points per vote round, which is fine for <100K.
fn pack_vote(dist_key: u32, local_idx: u32) -> i32 {
    return i32((dist_key << 16u) | (local_idx & 0xFFFFu));
}

fn unpack_vote_idx(vote: i32) -> u32 {
    return u32(vote) & 0xFFFFu;
}

@compute @workgroup_size(64)
fn vote_for_point(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_uninserted = params.x;

    if idx >= num_uninserted {
        return;
    }

    let vert_idx = uninserted[idx];
    let tet_idx = vert_tet[vert_idx];

    if tet_idx == INVALID {
        return;
    }
    if (tet_info[tet_idx] & TET_ALIVE) == 0u {
        return;
    }

    // Compute distance to tet centroid
    let tet = tets[tet_idx];
    let c = (points[tet.x].xyz + points[tet.y].xyz + points[tet.z].xyz + points[tet.w].xyz) * 0.25;
    let p = points[vert_idx].xyz;
    let d = p - c;
    let dist = dot(d, d);

    // Quantize distance to 16-bit key (smaller = closer = wins atomicMin)
    let dist_key = min(u32(dist * 65535.0), 65534u);

    let vote = pack_vote(dist_key, idx);
    atomicMin(&tet_vote[tet_idx], vote);
}

// Second pass: read the winning votes and mark the vert_tet mapping.
// Also outputs the list of (tet, vert) pairs to insert.

@group(1) @binding(0) var<storage, read> tets2: array<vec4<u32>>;
@group(1) @binding(1) var<storage, read> tet_info2: array<u32>;
@group(1) @binding(2) var<storage, read> tet_vote2: array<i32>;
@group(1) @binding(3) var<storage, read> uninserted2: array<u32>;
@group(1) @binding(4) var<storage, read_write> insert_list: array<vec2<u32>>; // (tet_idx, vert_idx)
@group(1) @binding(5) var<storage, read_write> counters: array<atomic<u32>>;
@group(1) @binding(6) var<uniform> params2: vec4<u32>; // x = max_tets

const COUNTER_INSERTED: u32 = 2u;

@compute @workgroup_size(64)
fn pick_winner_point(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let tet_idx = gid.x;
    let max_tets = params2.x;

    if tet_idx >= max_tets {
        return;
    }
    if (tet_info2[tet_idx] & TET_ALIVE) == 0u {
        return;
    }

    let vote = tet_vote2[tet_idx];
    if vote == NO_VOTE {
        return;
    }

    let local_idx = unpack_vote_idx(vote);
    let vert_idx = uninserted2[local_idx];

    // Append to insert list
    let slot = atomicAdd(&counters[COUNTER_INSERTED], 1u);
    insert_list[slot] = vec2<u32>(tet_idx, vert_idx);
}
