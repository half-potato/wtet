// Port of kerUpdateFlipTrace from gDel3D/GPU/KerDivision.cu (lines 741-785)
// Builds flip history chain for tet relocation
//
// CUDA code:
//   for ( int idx = getCurThreadIdx(); idx < flipNum; idx += getThreadNum() )
//   {
//       const int flipIdx = orgFlipNum + idx;
//       FlipItem flipItem = loadFlip( flipArr, flipIdx );
//
//       if ( flipItem._v[ 0 ] == -1 )   // All tets are empty, no need to trace
//           continue;
//
//       int tetIdx, nextFlip;
//
//       tetIdx              = flipItem._t[ 0 ];
//       nextFlip            = tetToFlip[ tetIdx ];
//       flipItem._t[ 0 ]    = ( nextFlip == -1 ) ? ( tetIdx << 1 ) | 0 : nextFlip;
//       tetToFlip[ tetIdx ] = ( flipIdx << 1 ) | 1;
//
//       tetIdx              = flipItem._t[ 1 ];
//       nextFlip            = tetToFlip[ tetIdx ];
//       flipItem._t[ 1 ]    = ( nextFlip == -1 ) ? ( tetIdx << 1 ) | 0 : nextFlip;
//       tetToFlip[ tetIdx ] = ( flipIdx << 1 ) | 1;
//
//       tetIdx              = flipItem._t[ 2 ];
//
//       if ( tetIdx < 0 )
//       {
//           tetIdx              = makePositive( tetIdx );
//           tetToFlip[ tetIdx ] = ( flipIdx << 1 ) | 1;
//       }
//       else
//       {
//           nextFlip            = tetToFlip[ tetIdx ];
//           flipItem._t[ 2 ]    = ( nextFlip == -1 ) ? ( tetIdx << 1 ) | 0 : nextFlip;
//       }
//
//       storeFlip( flipArr, flipIdx, flipItem );
//   }

@group(0) @binding(0) var<storage, read_write> flip_arr: array<vec4<i32>>; // FlipItem (3 tets + 1 vert)
@group(0) @binding(1) var<storage, read_write> tet_to_flip: array<i32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = org_flip_num, y = flip_num

fn make_positive(val: i32) -> u32 {
    if val < 0 {
        return u32(-val);
    }
    return u32(val);
}

@compute @workgroup_size(64)
fn update_flip_trace(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let org_flip_num = i32(params.x);
    let flip_num = params.y;

    if idx >= flip_num {
        return;
    }

    let flip_idx = org_flip_num + i32(idx);
    var flip_item = flip_arr[flip_idx];

    // If all tets are empty, no need to trace
    // FlipItem: _t[0], _t[1], _t[2], _v[0]
    // Check if first vertex is -1 (empty marker)
    if flip_item.w == -1 {
        return;
    }

    // Process first tet
    var tet_idx = flip_item.x;
    var next_flip = tet_to_flip[tet_idx];
    flip_item.x = select(next_flip, (tet_idx << 1) | 0, next_flip == -1);
    tet_to_flip[tet_idx] = (flip_idx << 1) | 1;

    // Process second tet
    tet_idx = flip_item.y;
    next_flip = tet_to_flip[tet_idx];
    flip_item.y = select(next_flip, (tet_idx << 1) | 0, next_flip == -1);
    tet_to_flip[tet_idx] = (flip_idx << 1) | 1;

    // Process third tet (may be negative for 2-3 flip)
    tet_idx = flip_item.z;

    if tet_idx < 0 {
        // Negative tet index
        let positive_idx = i32(make_positive(tet_idx));
        tet_to_flip[positive_idx] = (flip_idx << 1) | 1;
    } else {
        // Positive tet index
        next_flip = tet_to_flip[tet_idx];
        flip_item.z = select(next_flip, (tet_idx << 1) | 0, next_flip == -1);
        tet_to_flip[tet_idx] = (flip_idx << 1) | 1;
    }

    // Store updated flip item
    flip_arr[flip_idx] = flip_item;
}
