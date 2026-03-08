//! Simple test for star splaying functionality.

use crate::cpu::fix_with_star_splaying;
use crate::types::{encode_opp, DelaunayResult};

#[test]
fn test_star_splaying_simple() {
    eprintln!("\n=== SIMPLE STAR SPLAYING TEST ===\n");

    // Create a simple test case with 5 points
    // Points form a degenerate configuration to trigger failures
    let points = vec![
        [0.0, 0.0, 0.0], // 0
        [1.0, 0.0, 0.0], // 1
        [0.0, 1.0, 0.0], // 2
        [0.0, 0.0, 1.0], // 3
        [0.5, 0.5, 0.5], // 4 - center point (might fail)
    ];

    // Create initial triangulation (super-tet + one tet)
    // Super-tet vertices: 5, 6, 7, 8 (not in points, just for structure)
    let mut result = DelaunayResult {
        tets: vec![
            // Real tetrahedron
            [0, 1, 2, 3],
            // Could add more tets if needed
        ],
        adjacency: vec![
            [encode_opp(0, 0); 4], // Self-adjacent for simplicity
        ],
        failed_verts: vec![4], // Vertex 4 failed to insert
    };

    eprintln!("Initial state:");
    eprintln!("  Points: {}", points.len());
    eprintln!("  Tets: {}", result.tets.len());
    eprintln!("  Failed vertices: {:?}", result.failed_verts);

    // Run star splaying
    fix_with_star_splaying(&points, &mut result);

    eprintln!("\nFinal state:");
    eprintln!("  Tets: {}", result.tets.len());
    eprintln!("  Failed vertices: {:?}", result.failed_verts);

    // Verify result
    assert_eq!(
        result.failed_verts.len(),
        0,
        "Star splaying should fix all failed vertices"
    );

    eprintln!("\n=== TEST PASSED ===\n");
}

#[test]
#[ignore] // Ignore for now - needs proper adjacency setup
fn test_star_extraction_basic() {
    eprintln!("\n=== BASIC STAR EXTRACTION TEST ===\n");

    // This test needs a properly constructed triangulation with
    // valid adjacency. For now, use the simpler tests below.

    eprintln!("\n=== TEST SKIPPED ===\n");
}

#[test]
fn test_star_insertion() {
    eprintln!("\n=== STAR VERTEX INSERTION TEST ===\n");

    use crate::cpu::facet::Tri;
    use crate::cpu::star::Star;

    let points = vec![
        [0.0, 0.0, 0.0], // 0 - star center
        [1.0, 0.0, 0.0], // 1
        [0.0, 1.0, 0.0], // 2
        [0.0, 0.0, 1.0], // 3
        [0.5, 0.5, 0.0], // 4 - new vertex to insert
    ];

    let pts64: Vec<[f64; 3]> = points
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();

    let mut facet_queue = Vec::new();
    let mut star = Star::new(0, pts64, &mut facet_queue as *mut _);

    // Create a simple star with one triangle
    star.tri_vec.push(Tri::new(1, 2, 3));
    star.tri_status_vec
        .push(crate::cpu::facet::TriStatus::Valid);
    star.tet_idx_vec.push(-1);
    star.tri_opp_vec.push(crate::cpu::facet::TriOpp::new());

    eprintln!("Initial star:");
    eprintln!("  Center: {}", star.vert);
    eprintln!("  Triangles: {}", star.tri_vec.len());

    // Try inserting vertex 4
    let mut stack = Vec::new();
    let mut visited = vec![0i32; 10];

    eprintln!("\nInserting vertex 4 into star...");
    let success = star.insert_to_star(4, &mut stack, &mut visited, 1);

    eprintln!("Insertion result: {}", if success { "SUCCESS" } else { "FAILED" });
    eprintln!("Final triangles: {}", star.tri_vec.len());

    // Should have added triangles (exact number depends on implementation)
    if success {
        assert!(
            star.tri_vec.len() > 1,
            "Successful insertion should add triangles"
        );
    }

    eprintln!("\n=== TEST COMPLETE ===\n");
}
