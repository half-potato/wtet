// Port of kerRelocatePointsFast from gDel3D/GPU/KerPredicates.cu (lines 822-913, 916-930)
// Updates vert_tet for vertices whose containing tets were flipped
//
// Walks the flip trace chain (tetToFlip) to find the final tet location.
// For each vertex:
// 1. Follow chain of flips starting from current tet
// 2. Use orient3d tests to determine which of the new tets contains the vertex
// 3. Update vert_tet to point to the final tet

@group(0) @binding(0) var<storage, read> uninserted: array<u32>;
@group(0) @binding(1) var<storage, read_write> vert_tet: array<i32>;
@group(0) @binding(2) var<storage, read> tet_to_flip: array<i32>;
@group(0) @binding(3) var<storage, read> flip_arr: array<vec4<i32>>; // FlipItem = 2 × vec4<i32>
@group(0) @binding(4) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> params: vec4<u32>; // x = num_uninserted

fn remove_exact_bit(val: i32) -> i32 {
    return val & i32(~(1u << 31u));
}

fn orient3d_fast(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> i32 {
    let ad = a - d;
    let bd = b - d;
    let cd = c - d;
    let det = ad.x * (bd.y * cd.z - bd.z * cd.y)
            + bd.x * (cd.y * ad.z - cd.z * ad.y)
            + cd.x * (ad.y * bd.z - ad.z * bd.y);
    if det > 0.0 {
        return 1; // OrientPos
    } else if det < 0.0 {
        return -1; // OrientNeg
    }
    return 0; // OrientZero
}

fn load_flip_item(flip_idx: u32) -> array<i32, 8> {
    let v0 = flip_arr[flip_idx * 2u];
    let v1 = flip_arr[flip_idx * 2u + 1u];
    return array<i32, 8>(
        v0.x, v0.y, v0.z, v0.w,  // _v[0-3]
        v1.x,                     // _v[4]
        v1.y, v1.z, v1.w          // _t[0-2]
    );
}

@compute @workgroup_size(64)
fn relocate_points_fast(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let vert_idx = gid.x;
    let num_uninserted = params.x;

    if vert_idx >= num_uninserted {
        return;
    }

    let tet_idx_val = vert_tet[vert_idx];
    let tet_idx = remove_exact_bit(tet_idx_val);

    var next_idx = tet_to_flip[tet_idx];

    if next_idx == -1 {
        return; // Tet not flipped
    }

    let vertex = uninserted[vert_idx];
    let vertex_p = points[vertex].xyz;

    var flag = next_idx & 1;
    var dest_idx = next_idx >> 1;

    // Walk flip trace chain
    loop {
        if flag != 1 {
            break;
        }

        let flip_item = load_flip_item(u32(dest_idx));

        // Determine flip type: 3-2 if _t[2] < 0, else 2-3
        let f_type = select(0u, 1u, flip_item[7] < 0); // 0 = Flip23, 1 = Flip32

        var next_loc_id: i32;
        var F: vec3<u32>;

        // Pre-load flip_item vertices to avoid dynamic indexing
        let p0 = points[u32(flip_item[0])].xyz;
        let p1 = points[u32(flip_item[1])].xyz;
        let p2 = points[u32(flip_item[2])].xyz;
        let p3 = points[u32(flip_item[3])].xyz;
        let p4 = points[u32(flip_item[4])].xyz;

        var ord0: i32;
        if f_type == 0u {
            // Flip23: F = (0, 2, 3)
            ord0 = orient3d_fast(p0, p2, p3, vertex_p);
        } else {
            // Flip32: F = (0, 1, 2)
            ord0 = orient3d_fast(p0, p1, p2, vertex_p);
        }

        if ord0 == 0 {
            // Need exact computation - mark and break
            // For now, just keep current location
            break;
        }

        if f_type == 1u {
            // Flip32
            next_loc_id = select(1, 0, ord0 == 1);
        } else {
            // Flip23
            var ord1: i32;
            if ord0 == 1 {
                next_loc_id = 0;
                // F = (0, 3, 1)
                ord1 = orient3d_fast(p0, p3, p1, vertex_p);
            } else {
                next_loc_id = 1;
                // F = (0, 4, 3)
                ord1 = orient3d_fast(p0, p4, p3, vertex_p);
            }

            if ord1 == 0 {
                // Need exact computation - mark and break
                break;
            }

            next_loc_id = select(next_loc_id, 2, ord1 != 1);
        }

        next_idx = flip_item[5 + next_loc_id]; // _t[next_loc_id]
        flag = next_idx & 1;
        dest_idx = next_idx >> 1;
    }

    vert_tet[vert_idx] = dest_idx; // Write back final location
}
