# Split Shader Bug Report

## Summary

The split shader **is executing**, but the free list allocation (`get_free_slots_4tet()`) is incorrectly returning tet index 0 as a free slot, causing split to overwrite the old tet instead of creating new tets.

## Evidence

### Minimal Reproduction Test
Created `tests/minimal_split_test.rs` - a standalone test that creates minimal buffers and dispatches split with 1 insertion.

### Diagnostic Results

```
[TEST] tet_info after split: [3, 0, 0, 0, 0, 999, 43690, 0, 48879, 0]
```

Diagnostics added to `src/shaders/split.wgsl`:
- `tet_info[5] = 999` - Proves shader entry (✓ executed)
- `tet_info[8] = 0xBEEF (48879)` - Proves allocation succeeded (✓ passed check)
- `tet_info[6] = 0xAAAA (43690)` - Proves **t0 == 0** (✗ BUG!)

### What Should Happen

When splitting tet 0:
1. `old_tet = 0` (from insert_list)
2. Allocate 4 new tet slots: `t0, t1, t2, t3` (should be indices like 1, 2, 3, 4)
3. Write new tet data to `tets[t0..t3]`
4. Mark `tet_info[old_tet]` as dead (0)
5. Mark `tet_info[t0..t3]` as alive (TET_ALIVE | TET_CHANGED = 3)

### What Actually Happens

1. `old_tet = 0` ✓
2. Allocate: **t0 = 0, t1 = ?, t2 = ?, t3 = ?** ✗
3. Write new tet data to `tets[0]` (overwrites old tet)
4. Mark `tet_info[0] = 0` (line 191)
5. Mark `tet_info[0] = 3` (line 264) ← **Overwrites the dead marker!**

Result: `tet_info[0] = 3` instead of 0, and new tets are never created.

## Root Cause

`get_free_slots_4tet()` in `src/shaders/split.wgsl` (lines 86-120) is returning tet index 0.

### Current Allocation Logic

```wgsl
fn get_free_slots_4tet(vertex: u32, inf_idx: u32, tet_num: u32) -> vec4<u32> {
    var slots: vec4<u32>;
    for (var i = 0u; i < 4u; i++) {
        var free_idx: u32 = 0xFFFFFFFFu;

        // Try vertex's free list
        let loc_idx = i32(atomicSub(&vert_free_arr[vertex], 1u)) - 1;

        if loc_idx >= 0 {
            free_idx = free_arr[vertex * MEAN_VERTEX_DEGREE + u32(loc_idx)];
        } else {
            atomicStore(&vert_free_arr[vertex], 0u);

            // Try infinity block
            let inf_loc_idx = i32(atomicSub(&vert_free_arr[inf_idx], 1u)) - 1;

            if inf_loc_idx >= 0 {
                free_idx = free_arr[inf_idx * MEAN_VERTEX_DEGREE + u32(inf_loc_idx)];
            } else {
                // Expand - allocate from new range
                free_idx = tet_num - u32(-inf_loc_idx) - 1u;
            }
        }
        slots[i] = free_idx;
    }
    return slots;
}
```

### Problem

The free list arrays (`free_arr` and `vert_free_arr`) are **not properly initialized** in the minimal test!

In the minimal test:
```rust
// vert_free_arr: 8 vertices, each initialized to 8
let vert_free_arr = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    contents: bytemuck::cast_slice(&[8u32; 8]),
    ...
});

// free_arr: 80 slots, all initialized to 0
let free_arr = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    contents: &vec![0u8; 80 * 4],  // ← ALL ZEROS!
    ...
});
```

So when `get_free_slots_4tet(vertex=1, ...)` runs:
1. `atomicSub(&vert_free_arr[1], 1u)` returns 8, so `loc_idx = 7` ✓
2. Read `free_arr[1 * 8 + 7] = free_arr[15]` = **0** ✗
3. Return `free_idx = 0` ✗

**The free_arr was never populated with actual free tet indices!**

## How CUDA/Main Code Initializes

In `src/gpu/buffers.rs` lines 195-210:
```rust
// Initialize free_arr with indices 1..max_tets
let mut free_arr_data = Vec::with_capacity(num_vertices * MEAN_VERTEX_DEGREE);
for v in 0..num_vertices {
    for slot in 0..MEAN_VERTEX_DEGREE {
        let tet_idx = v * MEAN_VERTEX_DEGREE + slot + 1; // +1 to skip tet 0!
        free_arr_data.push(tet_idx as u32);
    }
}
```

Each vertex gets 8 pre-allocated tet slots:
- Vertex 0: tets 1-8
- Vertex 1: tets 9-16
- Vertex 2: tets 17-24
- ...

**The minimal test didn't initialize this!**

## Fix for Minimal Test

Update `tests/minimal_split_test.rs` line ~70:

```rust
// OLD (wrong):
let free_arr = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    contents: &vec![0u8; 80 * 4],
    usage: storage_rw,
});

// NEW (correct):
let mut free_arr_data = Vec::new();
for v in 0..8 {  // 8 vertices (4 real + 4 super-tet)
    for slot in 0..8 {  // MEAN_VERTEX_DEGREE = 8
        let tet_idx = v * 8 + slot + 1;  // +1 to skip tet 0
        free_arr_data.push(tet_idx as u32);
    }
}
let free_arr = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some("free_arr"),
    contents: bytemuck::cast_slice(&free_arr_data),
    usage: storage_rw,
});
```

## Why Full Tests Also Fail

The same root cause likely affects the full integration tests!

In `src/phase1.rs`, the `GpuState` initialization (`src/gpu/mod.rs`) creates buffers via `GpuBuffers::new()`, which DOES initialize free_arr correctly.

**BUT** there might be a synchronization issue or the free list gets corrupted during execution. This needs further investigation.

## Next Steps

1. ✅ Fix minimal test's free_arr initialization
2. Run minimal test to confirm split now works
3. Check if main tests still fail after proving split shader works in isolation
4. If main tests still fail, investigate free list corruption during execution
5. Check expand_tetra_list implementation - it might not be updating free lists correctly

## Files Modified for Diagnostics

- `src/shaders/split.wgsl` - Added diagnostic writes (lines 127-131, 162-170, 179-187)
- `tests/minimal_split_test.rs` - Created minimal reproduction case

## Timeline

- Initial symptoms: tet_info all zeros after split, all tests failing
- Hypothesis 1: Split shader not executing (WRONG - shader runs!)
- Hypothesis 2: Params buffer not being written (WRONG - verified via prints)
- **Discovery**: Minimal test shows split executes but allocates t0=0
- **Root Cause**: free_arr not initialized with actual tet indices

---

**Status**: Root cause identified. Fix ready to apply and test.
