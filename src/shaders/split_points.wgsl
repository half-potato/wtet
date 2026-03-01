// Port of kerSplitPointsFast from gDel3D/GPU/KerPredicates.cu (lines 210-283, 285-303)
// Updates vert_tet for uninserted vertices whose containing tets are being split
//
// Called AFTER mark_split and BEFORE split to ensure all uninserted vertices
// know which of the 4 new split tets they belong to.

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> uninserted: array<u32>; // vertexArr._arr
@group(0) @binding(2) var<storage, read_write> vert_tet: array<u32>; // vertexTetArr - position-indexed!
@group(0) @binding(3) var<storage, read> tet_to_vert: array<u32>; // which insertion is splitting each tet
@group(0) @binding(4) var<storage, read> tets: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> free_arr: array<u32>;
@group(0) @binding(6) var<storage, read> insert_list: array<vec2<u32>>; // to get split vertex
@group(0) @binding(7) var<uniform> params: vec4<u32>; // x = num_uninserted

const INVALID: u32 = 0xFFFFFFFFu;
const INT_MAX: u32 = 0xFFFFFFFFu;
const MEAN_VERTEX_DEGREE: u32 = 8u;

// From GPUDecl.h - decision tree for determining which split tet contains a vertex
// SplitFaces[face] gives indices into 5-vertex array (tet[0-3] + splitVertex at index 4)
const SPLIT_FACES: array<vec3<u32>, 11> = array<vec3<u32>, 11>(
    vec3<u32>(0u, 1u, 4u), // 0
    vec3<u32>(0u, 3u, 4u), // 1
    vec3<u32>(0u, 2u, 4u), // 2
    vec3<u32>(2u, 3u, 4u), // 3
    vec3<u32>(1u, 3u, 4u), // 4
    vec3<u32>(1u, 2u, 4u), // 5
    vec3<u32>(2u, 3u, 4u), // 6
    vec3<u32>(1u, 3u, 2u), // 7
    vec3<u32>(0u, 2u, 3u), // 8
    vec3<u32>(0u, 3u, 1u), // 9
    vec3<u32>(0u, 1u, 2u), // 10
);

// SplitNext[face][orient_result] - navigate decision tree
// [0] = positive orient, [1] = negative orient
const SPLIT_NEXT: array<vec2<u32>, 11> = array<vec2<u32>, 11>(
    vec2<u32>(1u, 2u),   // 0
    vec2<u32>(3u, 4u),   // 1
    vec2<u32>(5u, 6u),   // 2
    vec2<u32>(7u, 8u),   // 3
    vec2<u32>(9u, 7u),   // 4
    vec2<u32>(7u, 10u),  // 5
    vec2<u32>(7u, 8u),   // 6
    vec2<u32>(1u, 0u),   // 7
    vec2<u32>(2u, 0u),   // 8
    vec2<u32>(3u, 0u),   // 9
    vec2<u32>(4u, 0u),   // 10
);

fn orient3d_simple(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> f32 {
    let ad = a - d;
    let bd = b - d;
    let cd = c - d;
    return ad.x * (bd.y * cd.z - bd.z * cd.y)
         + bd.x * (cd.y * ad.z - cd.z * ad.y)
         + cd.x * (ad.y * bd.z - ad.z * bd.y);
}

@compute @workgroup_size(64)
fn split_points(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_uninserted = params.x;

    // Line 215: for ( int vertIdx = getCurThreadIdx(); vertIdx < vertexArr._num; ... )
    if idx >= num_uninserted {
        return;
    }

    // Line 217: int tetIdx = vertexTetArr[ vertIdx ];
    var tet_idx = vert_tet[idx];

    // Line 226: const int splitVertIdx = tetToVert[ tetIdx ];
    let split_vert_idx = tet_to_vert[tet_idx];

    // Line 230-234: if ( splitVertIdx == INT_MAX ) ... continue
    if split_vert_idx == INT_MAX {
        return; // Vertex's tetra will not be split in this round
    }

    // Lines 236-240: Get vertex and split vertex
    let vertex = uninserted[idx];
    let pt_vertex = points[vertex].xyz;

    // split_vert_idx is position in insert_list, get actual split vertex
    let split_position = insert_list[split_vert_idx].y;
    let split_vertex = uninserted[split_position];

    let tet = tets[tet_idx];

    // Line 242: const int freeIdx = ( splitVertex + 1 ) * MeanVertDegree - 1;
    let free_idx = (split_vertex + 1u) * MEAN_VERTEX_DEGREE - 1u;

    // Lines 243-250: Build 5-vertex array and get their positions
    let tet_vert = array<u32, 5>(tet.x, tet.y, tet.z, tet.w, split_vertex);
    let pt = array<vec3<f32>, 5>(
        points[tet_vert[0]].xyz,
        points[tet_vert[1]].xyz,
        points[tet_vert[2]].xyz,
        points[tet_vert[3]].xyz,
        points[tet_vert[4]].xyz,
    );

    // Lines 252-270: Decision tree navigation using orient3d
    var face = 0u;

    for (var i = 0u; i < 3u; i++) {
        let fv = SPLIT_FACES[face];

        // Lines 258-264: Orient3D test (Fast version only)
        let ort = orient3d_simple(
            pt[fv.x],
            pt[fv.y],
            pt[fv.z],
            pt_vertex
        );

        // Line 269: face = SplitNext[ face ][ ( ort == OrientPos ) ? 0 : 1 ];
        let next_idx = select(1u, 0u, ort > 0.0);
        face = SPLIT_NEXT[face][next_idx];
    }

    // Lines 272-277: face is now 7-10, map to actual tet index
    // face 7 → freeArr[freeIdx - 0]
    // face 8 → freeArr[freeIdx - 1]
    // face 9 → freeArr[freeIdx - 2]
    // face 10 → freeArr[freeIdx - 3]
    let new_tet_idx = free_arr[free_idx - (face - 7u)];

    // Line 279: vertexTetArr[ vertIdx ] = face;
    vert_tet[idx] = new_tet_idx;
}
