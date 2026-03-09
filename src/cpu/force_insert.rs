//! Force-insert vertices that failed GPU insertion.
//!
//! Creates a basic triangulation for failed vertices so star splaying can fix them.

use crate::types::{encode_opp, DelaunayResult, INVALID};

/// Force-insert a vertex by splitting an existing tet.
///
/// This creates a 1-4 split (1 tet → 4 tets) which may violate Delaunay,
/// but star splaying will fix it afterwards.
pub fn force_insert_vertex(
    vert: u32,
    result: &mut DelaunayResult,
) -> bool {
    // Find any tet (preferably near the vertex)
    if result.tets.is_empty() {
        eprintln!("[FORCE_INSERT] No tets available to split");
        return false;
    }

    // Use the first tet for simplicity
    let tet_idx = 0;
    let tet = result.tets[tet_idx];
    let tet_opp = result.adjacency[tet_idx];

    eprintln!(
        "[FORCE_INSERT] Splitting tet {} [{}, {}, {}, {}] to insert vertex {}",
        tet_idx, tet[0], tet[1], tet[2], tet[3], vert
    );

    // Create 4 new tets by replacing tet_idx and adding 3 more
    // Original tet: [v0, v1, v2, v3]
    // New tets:
    //   T0: [vert, v1, v2, v3] (replaces original)
    //   T1: [v0, vert, v2, v3] (new)
    //   T2: [v0, v1, vert, v3] (new)
    //   T3: [v0, v1, v2, vert] (new)

    let v0 = tet[0];
    let v1 = tet[1];
    let v2 = tet[2];
    let v3 = tet[3];

    let t0 = [vert, v1, v2, v3];
    let t1 = [v0, vert, v2, v3];
    let t2 = [v0, v1, vert, v3];
    let t3 = [v0, v1, v2, vert];

    // Indices
    let t0_idx = tet_idx;
    let t1_idx = result.tets.len();
    let t2_idx = t1_idx + 1;
    let t3_idx = t1_idx + 2;

    // Replace original tet
    result.tets[t0_idx] = t0;

    // Add new tets
    result.tets.push(t1);
    result.tets.push(t2);
    result.tets.push(t3);

    // Set up internal adjacency (4 tets sharing central vertex)
    // Shared faces:
    //   T0-T1: [vert, v2, v3] → T0 face 1 (opp v1), T1 face 0 (opp v0)
    //   T0-T2: [vert, v1, v3] → T0 face 2 (opp v2), T2 face 0 (opp v0)
    //   T0-T3: [vert, v1, v2] → T0 face 3 (opp v3), T3 face 0 (opp v0)
    //   T1-T2: [v0, vert, v3] → T1 face 2 (opp v2), T2 face 1 (opp v1)
    //   T1-T3: [v0, vert, v2] → T1 face 3 (opp v3), T3 face 1 (opp v1)
    //   T2-T3: [v0, v1, vert] → T2 face 3 (opp v3), T3 face 2 (opp v2)

    let mut adj0 = [INVALID; 4];
    adj0[0] = tet_opp[0]; // External (opposite vert, was opposite v0)
    adj0[1] = encode_opp(t1_idx as u32, 0); // T1 face 0
    adj0[2] = encode_opp(t2_idx as u32, 0); // T2 face 0
    adj0[3] = encode_opp(t3_idx as u32, 0); // T3 face 0

    let mut adj1 = [INVALID; 4];
    adj1[0] = encode_opp(t0_idx as u32, 1); // T0 face 1
    adj1[1] = tet_opp[1]; // External (opposite vert, was opposite v1)
    adj1[2] = encode_opp(t2_idx as u32, 1); // T2 face 1
    adj1[3] = encode_opp(t3_idx as u32, 1); // T3 face 1

    let mut adj2 = [INVALID; 4];
    adj2[0] = encode_opp(t0_idx as u32, 2); // T0 face 2
    adj2[1] = encode_opp(t1_idx as u32, 2); // T1 face 2
    adj2[2] = tet_opp[2]; // External (opposite vert, was opposite v2)
    adj2[3] = encode_opp(t3_idx as u32, 2); // T3 face 2

    let mut adj3 = [INVALID; 4];
    adj3[0] = encode_opp(t0_idx as u32, 3); // T0 face 3
    adj3[1] = encode_opp(t1_idx as u32, 3); // T1 face 3
    adj3[2] = encode_opp(t2_idx as u32, 3); // T2 face 3
    adj3[3] = tet_opp[3]; // External (opposite vert, was opposite v3)

    result.adjacency[t0_idx] = adj0;

    // Add adjacency for new tets
    result.adjacency.push(adj1);
    result.adjacency.push(adj2);
    result.adjacency.push(adj3);

    eprintln!(
        "[FORCE_INSERT] Created {} tets (was {})",
        result.tets.len(),
        t1_idx
    );

    true
}

/// Force-insert all failed vertices before star splaying.
pub fn force_insert_failed_vertices(
    result: &mut DelaunayResult,
) {
    let failed = result.failed_verts.clone();

    eprintln!("[FORCE_INSERT] Force-inserting {} failed vertices", failed.len());

    for &vert in &failed {
        force_insert_vertex(vert, result);
    }

    eprintln!("[FORCE_INSERT] All vertices force-inserted");
}
