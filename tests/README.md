# Isolated 4-Tet Allocation Test

## Purpose

Debugging the 4-tet allocation bug in a minimal isolated environment, WITHOUT the full Delaunay triangulation complexity.

## Status

✅ **`test_minimal_4tet_allocation` - PASSES**
- Tests ONLY the allocation function `get_free_slots_4tet`
- Verifies correct indices returned: [64, 63, 62, 61]
- Verifies `vert_free_arr` decremented correctly

**Conclusion:** The allocation logic itself works perfectly!

## What This Tells Us

The bug is NOT in the allocation, but in how the allocated tets are USED:
- Writing new tet data?
- Updating adjacency?
- Marking old tet dead?
- Updating counters?
- Something else in the split kernel?

## How to Use This

### Run the passing test:
```bash
cargo test --test test_4tet_allocation test_minimal_4tet_allocation
```

### Expand to find the bug:

1. **Add more steps incrementally:**
   ```rust
   // Start with: allocate only (PASSES)
   let slots = get_free_slots_4tet(0u);

   // Add: write new tets
   // Does it still pass?

   // Add: update adjacency
   // Does it still pass?

   // Keep adding until it FAILS
   ```

2. **Test the variable naming hypothesis:**
   ```wgsl
   // Test A: Use old_tet variable
   let old_tet = insert.x;
   let slots = get_free_slots_4tet(p);
   let t0 = slots.x;
   // ...

   // Test B: Keep t0 name
   let t0_original = insert.x;
   let slots = get_free_slots_4tet(p);
   let t0 = slots.x;  // Reuse t0 name
   // ...

   // Compare results
   ```

3. **Add debugging output:**
   ```wgsl
   // Count how many splits attempt vs succeed
   atomicAdd(&debug_counters[0], 1u);  // Attempted
   // ... allocation code ...
   atomicAdd(&debug_counters[1], 1u);  // After allocation
   // ... write tets ...
   atomicAdd(&debug_counters[2], 1u);  // After write
   ```

## Next Steps

1. Implement `test_minimal_split_with_4tet`:
   - Copy full split kernel logic
   - Run with minimal input (single tet split)
   - Add steps incrementally until failure point found

2. Implement `test_variable_naming`:
   - Test if using `old_tet` vs `t0` matters
   - Compare outputs

3. Once failure point found:
   - Fix it!
   - Test with full pipeline
   - Should unblock the entire cascade

## Files

- `test_4tet_allocation.rs` - This test file
- `README.md` - This file
- `../STATUS.md` - Overall bug status
- `../TODO.md` - What needs to be done
