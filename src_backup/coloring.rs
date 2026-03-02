//! Graph coloring for tetrahedral meshes.
//!
//! Colors a set of tets such that no two adjacent tets have the same color.
//! Uses a simple greedy coloring algorithm.

use std::collections::HashSet;

/// Color a set of tets represented by their adjacency lists.
/// Returns a vec of (tet_idx, color) pairs.
pub fn color_tets(
    tet_indices: &[u32],
    tet_adjacency: &[Vec<u32>], // adjacency[i] = list of neighbors of tet_indices[i]
    max_colors: u32,
) -> Vec<u32> {
    let n = tet_indices.len();
    let mut colors = vec![u32::MAX; n];

    // Build index map for quick lookup
    let mut tet_to_idx = std::collections::HashMap::new();
    for (i, &tet) in tet_indices.iter().enumerate() {
        tet_to_idx.insert(tet, i);
    }

    // Greedy coloring
    for i in 0..n {
        let mut used_colors = HashSet::new();

        // Check colors of already-colored neighbors
        for &neighbor_tet in &tet_adjacency[i] {
            if let Some(&neighbor_idx) = tet_to_idx.get(&neighbor_tet) {
                if neighbor_idx < i && colors[neighbor_idx] != u32::MAX {
                    used_colors.insert(colors[neighbor_idx]);
                }
            }
        }

        // Find smallest unused color
        let mut color = 0;
        while color < max_colors && used_colors.contains(&color) {
            color += 1;
        }

        colors[i] = color;
    }

    colors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_coloring() {
        // Triangle: 0-1, 1-2, 2-0 (requires 3 colors)
        let tets = vec![0, 1, 2];
        let adj = vec![
            vec![1, 2],  // 0 neighbors 1,2
            vec![0, 2],  // 1 neighbors 0,2
            vec![0, 1],  // 2 neighbors 0,1
        ];

        let colors = color_tets(&tets, &adj, 4);
        
        // Verify no two adjacent have same color
        assert_ne!(colors[0], colors[1]);
        assert_ne!(colors[1], colors[2]);
        assert_ne!(colors[2], colors[0]);
    }

    #[test]
    fn test_independent_set() {
        // Two independent tets (no edges)
        let tets = vec![0, 1];
        let adj = vec![vec![], vec![]];

        let colors = color_tets(&tets, &adj, 4);
        
        // Can both be color 0 since independent
        assert_eq!(colors[0], 0);
        assert_eq!(colors[1], 0);
    }
}
