//! CPU-side algorithms for fixing Delaunay violations via star splaying.
//!
//! This module implements the CUDA star splaying algorithm from:
//! gDel3D/GDelFlipping/src/gDel3D/CPU/
//!
//! The algorithm extracts stars (links) around failed vertices, performs
//! local 2D Delaunay flipping on the link manifold, and reintegrates them
//! back into the 3D triangulation.

pub mod facet;
pub mod star;
pub mod splaying;
pub mod force_insert;

#[cfg(test)]
mod test_splaying;

// Re-export main entry point
pub use splaying::fix_with_star_splaying;
