# gDel3D Port - File Structure Plan

## File Correspondence

### Original → New Mapping

```
gDel3D/GDelFlipping/src/gDel3D/
├── GpuDelaunay.cu          → src/gpu_delaunay.rs
├── GPU/
│   ├── KerDivision.cu      → src/shaders/ker_division.wgsl
│   ├── KerPredicates.cu    → src/shaders/ker_predicates.wgsl
│   └── ThrustWrapper.cu    → src/thrust_wrapper.rs (Rust sorting/compact)
```

## Kernel Inventory

### KerDivision.cu (~24 kernels)
Primary kernels to port:
1. kerMakeFirstTetra - Initialize super-tetrahedron
2. kerSplitTetra - Main 1-to-4 split operation
3. kerMarkRejectedFlips - Flip conflict detection
4. kerFlip* - Various flip operations (2-to-3, 3-to-2, etc.)
5. kerUpdateOpp - Update adjacency information
6. Helper kernels for compaction/cleanup

### KerPredicates.cu
Geometric predicates:
1. kerPointLocationFast - Fast point location walk
2. kerPointLocationExact - Exact arithmetic version
3. Orient3D/InSphere predicates
4. Sphere computation kernels

### Host-side (GpuDelaunay.cu → gpu_delaunay.rs)
Orchestration functions:
1. compute() - Main entry point
2. insertPoints() - Insertion iteration loop
3. doFlipping() - Flip iteration loop
4. expandTetraList() - Buffer management
5. Helper functions for buffer operations

## Implementation Strategy

### Phase 1: Core Data Structures
- [ ] Port Tet/TetOpp structures
- [ ] Port encoding/decoding helpers
- [ ] Set up buffer layouts to match original

### Phase 2: Initialization
- [ ] Port kerMakeFirstTetra
- [ ] Test super-tet creation

### Phase 3: Point Location
- [ ] Port kerPointLocationFast
- [ ] Test with simple cases

### Phase 4: Splitting (Critical)
- [ ] Port kerSplitTetra with all adjacency logic
- [ ] Port expansion/allocation logic
- [ ] Test with 4-point case

### Phase 5: Flipping
- [ ] Port flip detection kernels
- [ ] Port flip execution kernels
- [ ] Test Delaunay property

### Phase 6: Host Orchestration
- [ ] Port main loop logic
- [ ] Port buffer management
- [ ] Full integration test

## Key Principles

1. **Algorithm Fidelity**: Study original carefully, implement same logic
2. **Fresh Implementation**: Write new WGSL/Rust code, don't copy-paste
3. **Preserve Structure**: Same file organization, function correspondence
4. **Add Safety**: WGSL validation helps catch errors original might miss
5. **Document Differences**: Note where WGSL requires different approaches

## Next Steps

1. Backup current implementation to `src_backup/`
2. Create new file structure
3. Implement phase by phase, testing each
4. Keep tests running throughout

Ready to proceed?
