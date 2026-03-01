// Port of kerMakeFirstTetra from gDel3D/GPU/KerDivision.cu (lines 46-95)
// This creates the initial super-tetrahedron structure

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> tets: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> tet_opp: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> tet_info: array<u32>;
@group(0) @binding(4) var<storage, read_write> vert_tet: array<u32>;
@group(0) @binding(5) var<storage, read_write> free_arr: array<u32>;
@group(0) @binding(6) var<storage, read_write> vert_free_arr: array<u32>;
@group(0) @binding(7) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> params: vec4<u32>; // x = num_points, y = max_tets

const INVALID: u32 = 0xFFFFFFFFu;
const TET_ALIVE: u32 = 1u;

// Helper for TetOpp encoding (from CommonTypes.h line 266)
fn make_opp_val(tet_idx: u32, opp_tet_vi: u32) -> u32 {
    return (tet_idx << 5u) | opp_tet_vi;
}

// Direct port of kerMakeFirstTetra (lines 46-95)
// Original creates 5 tets forming the initial super-tetrahedron
@compute @workgroup_size(1)
fn make_first_tetra() {
    let n = params.x;
    let max_tets = params.y;

    // The super-tetrahedron uses 4 virtual vertices at indices n, n+1, n+2, n+3
    // Original line 52: Tet tet parameter defines the first tet
    // We use the same vertex ordering as original
    let inf_idx = n; // In original this is passed as infIdx parameter

    // Lines 57-63: Define the 5 tets
    // tets[0] = { tet._v[0], tet._v[1], tet._v[2], tet._v[3] }
    // For super-tet, this is the 4 virtual vertices
    // Based on typical super-tet construction: use all 4 infinity vertices
    let v0 = n;
    let v1 = n + 1u;
    let v2 = n + 2u;
    let v3 = n + 3u;

    // Lines 57-79: The 5 tets and their adjacency
    // tet 0: (v0, v1, v2, v3)
    // tet 1: (v1, v2, v3, infIdx)
    // tet 2: (v0, v3, v2, infIdx)
    // tet 3: (v0, v1, v3, infIdx)
    // tet 4: (v0, v2, v1, infIdx)
    //
    // For simplicity, we create a single super-tet as before
    // (The original's 5-tet structure is for a specific configuration)
    // Let's match the original structure exactly:

    let tet_idx = 0u; // Original passes tetIdx parameter, we use 0

    // Lines 57-63: tets array
    tets[tet_idx + 0u] = vec4<u32>(v0, v1, v2, v3);
    tets[tet_idx + 1u] = vec4<u32>(v1, v2, v3, inf_idx);
    tets[tet_idx + 2u] = vec4<u32>(v0, v3, v2, inf_idx);
    tets[tet_idx + 3u] = vec4<u32>(v0, v1, v3, inf_idx);
    tets[tet_idx + 4u] = vec4<u32>(v0, v2, v1, inf_idx);

    // Lines 65-79: oppTet and oppVi arrays define adjacency structure
    // const int oppTet[][4] = {
    //     { 1, 2, 3, 4 },
    //     { 2, 3, 4, 0 },
    //     { 1, 4, 3, 0 },
    //     { 1, 2, 4, 0 },
    //     { 1, 3, 2, 0 }
    // };
    // const int oppVi[][4] = {
    //     { 3, 3, 3, 3 },
    //     { 0, 0, 0, 0 },
    //     { 0, 2, 1, 1 },
    //     { 1, 2, 1, 2 },
    //     { 2, 2, 1, 3 }
    // };

    // Lines 81-93: Set adjacency for all 5 tets
    // tet 0
    atomicStore(&tet_opp[(tet_idx + 0u) * 4u + 0u], make_opp_val(tet_idx + 1u, 3u));
    atomicStore(&tet_opp[(tet_idx + 0u) * 4u + 1u], make_opp_val(tet_idx + 2u, 3u));
    atomicStore(&tet_opp[(tet_idx + 0u) * 4u + 2u], make_opp_val(tet_idx + 3u, 3u));
    atomicStore(&tet_opp[(tet_idx + 0u) * 4u + 3u], make_opp_val(tet_idx + 4u, 3u));
    tet_info[tet_idx + 0u] = TET_ALIVE;

    // tet 1
    atomicStore(&tet_opp[(tet_idx + 1u) * 4u + 0u], make_opp_val(tet_idx + 2u, 0u));
    atomicStore(&tet_opp[(tet_idx + 1u) * 4u + 1u], make_opp_val(tet_idx + 3u, 0u));
    atomicStore(&tet_opp[(tet_idx + 1u) * 4u + 2u], make_opp_val(tet_idx + 4u, 0u));
    atomicStore(&tet_opp[(tet_idx + 1u) * 4u + 3u], make_opp_val(tet_idx + 0u, 0u));
    tet_info[tet_idx + 1u] = TET_ALIVE;

    // tet 2
    atomicStore(&tet_opp[(tet_idx + 2u) * 4u + 0u], make_opp_val(tet_idx + 1u, 0u));
    atomicStore(&tet_opp[(tet_idx + 2u) * 4u + 1u], make_opp_val(tet_idx + 4u, 2u));
    atomicStore(&tet_opp[(tet_idx + 2u) * 4u + 2u], make_opp_val(tet_idx + 3u, 1u));
    atomicStore(&tet_opp[(tet_idx + 2u) * 4u + 3u], make_opp_val(tet_idx + 0u, 1u));
    tet_info[tet_idx + 2u] = TET_ALIVE;

    // tet 3
    atomicStore(&tet_opp[(tet_idx + 3u) * 4u + 0u], make_opp_val(tet_idx + 1u, 1u));
    atomicStore(&tet_opp[(tet_idx + 3u) * 4u + 1u], make_opp_val(tet_idx + 2u, 2u));
    atomicStore(&tet_opp[(tet_idx + 3u) * 4u + 2u], make_opp_val(tet_idx + 4u, 1u));
    atomicStore(&tet_opp[(tet_idx + 3u) * 4u + 3u], make_opp_val(tet_idx + 0u, 2u));
    tet_info[tet_idx + 3u] = TET_ALIVE;

    // tet 4
    atomicStore(&tet_opp[(tet_idx + 4u) * 4u + 0u], make_opp_val(tet_idx + 1u, 2u));
    atomicStore(&tet_opp[(tet_idx + 4u) * 4u + 1u], make_opp_val(tet_idx + 3u, 2u));
    atomicStore(&tet_opp[(tet_idx + 4u) * 4u + 2u], make_opp_val(tet_idx + 2u, 1u));
    atomicStore(&tet_opp[(tet_idx + 4u) * 4u + 3u], make_opp_val(tet_idx + 0u, 3u));
    tet_info[tet_idx + 4u] = TET_ALIVE;

    // All input points start in tet 0
    for (var i = 0u; i < n; i++) {
        vert_tet[i] = tet_idx;
    }

    // Set counters
    atomicStore(&counters[0], max_tets - 5u); // free_count (5 tets used)
    atomicStore(&counters[1], 5u);            // active_count = 5
    atomicStore(&counters[2], 0u);            // inserted_count
    atomicStore(&counters[3], 0u);            // failed_count
}
