// Kernel: Pick the winning point from each tet's vote.
//
// Second pass of the vote system. Each tet that received a vote reads the
// winner and appends a (tet_idx, vert_idx) pair to the insert list.

@group(0) @binding(0) var<storage, read> tets: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> tet_info: array<u32>;
@group(0) @binding(2) var<storage, read> tet_vote: array<i32>;
@group(0) @binding(3) var<storage, read> uninserted: array<u32>;
@group(0) @binding(4) var<storage, read_write> insert_list: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params: vec4<u32>; // x = max_tets

const INVALID: u32 = 0xFFFFFFFFu;
const NO_VOTE: i32 = 0x7FFFFFFF;
const TET_ALIVE: u32 = 1u;
const COUNTER_INSERTED: u32 = 2u;

fn unpack_vote_idx(vote: i32) -> u32 {
    return u32(vote) & 0xFFFFu;
}

@compute @workgroup_size(64)
fn pick_winner_point(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let tet_idx = gid.x;
    let max_tets = params.x;

    if tet_idx >= max_tets {
        return;
    }
    if (tet_info[tet_idx] & TET_ALIVE) == 0u {
        return;
    }

    let vote = tet_vote[tet_idx];
    if vote == NO_VOTE {
        return;
    }

    let local_idx = unpack_vote_idx(vote);
    let vert_idx = uninserted[local_idx];

    // Append to insert list
    let slot = atomicAdd(&counters[COUNTER_INSERTED], 1u);
    insert_list[slot] = vec2<u32>(tet_idx, vert_idx);
}
