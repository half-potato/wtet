# GPU Debugging Tools: Brainstorm

## The Problem

Current debugging limitations:
- ❌ Can't inspect shader variables during execution
- ❌ Can't trace which exact instruction crashes
- ❌ Can't single-step through shader code
- ❌ No stack traces or line numbers for GPU crashes
- ❌ Can't see memory state at crash time
- ❌ Debug counters are coarse-grained and limited

## Category 1: Value Inspection Tools

### 1.1 Trace Buffer (High Priority, Medium Effort)

**Concept:** Dedicated buffer for logging shader values

```wgsl
@group(0) @binding(N) var<storage, read_write> trace_buffer: array<u32>;
@group(0) @binding(N+1) var<storage, read_write> trace_index: atomic<u32>;

fn trace_u32(value: u32) {
    let idx = atomicAdd(&trace_index, 1u);
    if idx < MAX_TRACE_ENTRIES {
        trace_buffer[idx] = value;
    }
}

fn trace_vec4(v: vec4<u32>) {
    let idx = atomicAdd(&trace_index, 4u);
    if idx + 3u < MAX_TRACE_ENTRIES {
        trace_buffer[idx] = v.x;
        trace_buffer[idx + 1u] = v.y;
        trace_buffer[idx + 2u] = v.z;
        trace_buffer[idx + 3u] = v.w;
    }
}

// Usage:
trace_u32(old_tet);  // Log: old_tet = ?
trace_vec4(new_slots);  // Log: t0, t1, t2, t3
```

**Readback format:**
```rust
struct TraceEntry {
    thread_id: u32,
    line_number: u32,
    value: u32,
}
```

**Pros:**
- Can log ANY value at ANY point
- Captures execution order
- Can log from multiple threads

**Cons:**
- Large buffer needed (millions of entries)
- Parsing output tedious
- Atomics add overhead

**Implementation:**
1. Add trace buffer to GpuBuffers
2. Macro for easy tracing: `TRACE!(old_tet, t0, t1)`
3. Read back and pretty-print results

### 1.2 Per-Thread Debug Slots (Medium Priority, Low Effort)

**Concept:** Each thread gets dedicated debug slots (no atomics needed)

```wgsl
@group(0) @binding(N) var<storage, read_write> thread_debug: array<vec4<u32>>;

fn debug_checkpoint(thread_id: u32, slot: u32, values: vec4<u32>) {
    thread_debug[thread_id * 16u + slot] = values;
}

// Usage in split kernel:
let tid = gid.x;
debug_checkpoint(tid, 0u, vec4<u32>(old_tet, p, 0u, 0u));  // Entry
debug_checkpoint(tid, 1u, new_slots);  // After allocation
debug_checkpoint(tid, 2u, vec4<u32>(t0, t1, t2, t3));  // Before write
debug_checkpoint(tid, 3u, vec4<u32>(999u, 999u, 999u, 999u));  // After write
```

**Pros:**
- No atomics, no races
- Fast, minimal overhead
- Easy to inspect specific thread

**Cons:**
- Large buffer (max_threads * 16 * 16 bytes)
- Only works for bounded thread counts

### 1.3 Conditional Breakpoint Buffer (Low Priority, High Value)

**Concept:** Log ONLY when interesting condition met

```wgsl
fn trace_if(condition: bool, marker: u32, values: vec4<u32>) {
    if condition {
        let idx = atomicAdd(&trace_index, 5u);
        trace_buffer[idx] = marker;  // e.g., 0xDEADBEEF
        trace_buffer[idx + 1u] = values.x;
        trace_buffer[idx + 2u] = values.y;
        trace_buffer[idx + 3u] = values.z;
        trace_buffer[idx + 4u] = values.w;
    }
}

// Usage:
trace_if(old_tet == 0u, 0xDEADBEEF, vec4<u32>(old_tet, p, t0, t1));
trace_if(t0 >= 512u, 0xBADBAD, new_slots);
```

**Pros:**
- Minimal overhead (only logs when triggered)
- Captures rare edge cases

**Cons:**
- Need to know what condition to check

## Category 2: Execution Flow Tracers

### 2.1 Breadcrumb Trail (High Priority, Low Effort)

**Concept:** Each thread marks progress through code

```wgsl
const BREADCRUMB_START = 1u;
const BREADCRUMB_ALLOC = 2u;
const BREADCRUMB_WRITE_T0 = 3u;
const BREADCRUMB_WRITE_ALL = 4u;
const BREADCRUMB_MARK_DEAD = 5u;
const BREADCRUMB_ADJACENCY = 6u;
const BREADCRUMB_COMPLETE = 99u;

@group(0) @binding(N) var<storage, read_write> breadcrumbs: array<u32>;

fn mark_breadcrumb(thread_id: u32, crumb: u32) {
    breadcrumbs[thread_id] = crumb;
}

// Usage:
mark_breadcrumb(gid.x, BREADCRUMB_START);
// ... allocation code ...
mark_breadcrumb(gid.x, BREADCRUMB_ALLOC);
tets[t0] = ...;
mark_breadcrumb(gid.x, BREADCRUMB_WRITE_T0);
// ... more writes ...
mark_breadcrumb(gid.x, BREADCRUMB_COMPLETE);
```

**After crash, read back breadcrumbs:**
```rust
for (thread_id, crumb) in breadcrumbs.iter().enumerate() {
    if *crumb != BREADCRUMB_COMPLETE {
        eprintln!("Thread {} crashed at breadcrumb {}", thread_id, crumb);
    }
}
```

**Pros:**
- Instantly shows WHERE crash happened
- Very low overhead (single write per checkpoint)
- Easy to implement

**Cons:**
- Only shows coarse progress, not exact line

**Enhancement:** Two-level breadcrumbs:
```wgsl
breadcrumbs[thread_id * 2] = major_breadcrumb;
breadcrumbs[thread_id * 2 + 1] = line_number;
```

### 2.2 Call Stack Tracker (Low Priority, Medium Effort)

**Concept:** Track function call depth and path

```wgsl
var<private> call_stack: array<u32, 16>;
var<private> stack_depth: u32 = 0u;

fn push_call(function_id: u32) {
    call_stack[stack_depth] = function_id;
    stack_depth += 1u;
}

fn pop_call() {
    stack_depth -= 1u;
}

// In get_free_slots_4tet:
push_call(1u);  // 1 = get_free_slots_4tet
// ... function body ...
pop_call();

// On crash, write stack to trace buffer
fn dump_stack() {
    for (var i = 0u; i < stack_depth; i++) {
        trace_u32(call_stack[i]);
    }
}
```

### 2.3 Hot Path Heatmap (Medium Priority, Medium Effort)

**Concept:** Count how many times each code path executes

```wgsl
@group(0) @binding(N) var<storage, read_write> heatmap: array<atomic<u32>>;

const HEATMAP_ALLOC_SUCCESS = 0u;
const HEATMAP_ALLOC_FAIL = 1u;
const HEATMAP_EARLY_RETURN = 2u;
// ... etc

fn heat(location: u32) {
    atomicAdd(&heatmap[location], 1u);
}

// Usage:
if count < 4u {
    heat(HEATMAP_ALLOC_FAIL);
    return INVALID;
}
heat(HEATMAP_ALLOC_SUCCESS);
```

**Pros:**
- Shows which branches are taken
- Reveals unexpected code paths

## Category 3: Memory Inspection Tools

### 3.1 Memory Snapshot Tool (High Priority, High Effort)

**Concept:** Capture buffer state before/after operations

```rust
// In test harness
async fn snapshot_buffers(&self, label: &str) -> BufferSnapshot {
    BufferSnapshot {
        label: label.to_string(),
        tets: self.read_buffer(&self.buffers.tets, max_tets).await,
        tet_info: self.read_buffer(&self.buffers.tet_info, max_tets).await,
        tet_opp: self.read_buffer(&self.buffers.tet_opp, max_tets * 4).await,
        counters: self.read_counters().await,
    }
}

// Usage:
let before = state.snapshot_buffers("before_split").await;
pollster::block_on(dispatch_split(...));
let after = state.snapshot_buffers("after_split").await;

let diff = compare_snapshots(&before, &after);
diff.print_changes();
```

**Output:**
```
Changes from before_split -> after_split:
  tets[0]: (4, 5, 7, 6) -> (4, 1, 2, 3)  CHANGED
  tets[320]: (0, 0, 0, 0) -> (4, 1, 2, 3)  NEW
  tet_info[0]: 1 -> 0  MARKED_DEAD
  counters.active_count: 1 -> 4  +3
```

**Pros:**
- See EXACTLY what changed
- Compare expected vs actual
- Catch corruption

**Cons:**
- Slow (full buffer readback)
- Large memory footprint

### 3.2 Watch Points (Medium Priority, High Effort)

**Concept:** Detect when specific memory location changes

```wgsl
@group(0) @binding(N) var<storage, read_write> watchpoints: array<u32>;
// watchpoints[i * 3 + 0] = watched_address
// watchpoints[i * 3 + 1] = expected_value
// watchpoints[i * 3 + 2] = triggered (0 or 1)

fn check_watchpoint(buffer_id: u32, index: u32, old_val: u32, new_val: u32) {
    for (var i = 0u; i < NUM_WATCHPOINTS; i++) {
        let addr = watchpoints[i * 3u];
        let expected = watchpoints[i * 3u + 1u];
        if addr == (buffer_id << 24u | index) && new_val != expected {
            watchpoints[i * 3u + 2u] = 1u;  // Triggered!
        }
    }
}

// Before any write to monitored buffer:
check_watchpoint(BUFFER_TETS, t0, 0u, vec4_to_u32(new_value));
```

**Pros:**
- Catch unexpected writes
- Detect race conditions

**Cons:**
- Significant overhead
- Hard to implement for all writes

### 3.3 Buffer Validator (High Priority, Medium Effort)

**Concept:** Dedicated kernel to check buffer consistency

```wgsl
// validate_buffers.wgsl
@compute @workgroup_size(64)
fn validate_tets() {
    let tet_idx = gid.x;
    if tet_idx >= max_tets { return; }

    let info = tet_info[tet_idx];
    if (info & TET_ALIVE) == 0u { return; }

    let tet = tets[tet_idx];

    // Check: All vertices valid
    if tet.x >= num_vertices || tet.y >= num_vertices ||
       tet.z >= num_vertices || tet.w >= num_vertices {
        atomicAdd(&errors[ERROR_INVALID_VERTEX], 1u);
    }

    // Check: No duplicate vertices
    if tet.x == tet.y || tet.x == tet.z || tet.x == tet.w ||
       tet.y == tet.z || tet.y == tet.w || tet.z == tet.w {
        atomicAdd(&errors[ERROR_DUPLICATE_VERTEX], 1u);
    }

    // Check: Adjacency back-pointers
    for (var face = 0u; face < 4u; face++) {
        let opp = get_opp(tet_idx, face);
        if opp != INVALID {
            let nei_tet = decode_opp_tet(opp);
            let nei_face = decode_opp_face(opp);

            if (tet_info[nei_tet] & TET_ALIVE) == 0u {
                atomicAdd(&errors[ERROR_DEAD_NEIGHBOR], 1u);
            }

            let back = get_opp(nei_tet, nei_face);
            if back != encode_opp(tet_idx, face) {
                atomicAdd(&errors[ERROR_BROKEN_BACKPOINTER], 1u);
            }
        }
    }
}
```

**Usage:**
```rust
// After each major operation
state.dispatch_validate(&mut encoder);
let errors = state.read_errors().await;
if errors.has_any() {
    panic!("Validation failed: {:?}", errors);
}
```

**Pros:**
- Catch corruption immediately after it happens
- Comprehensive checks
- Reusable across tests

## Category 4: Automated Comparison Tools

### 4.1 Reference Implementation Diff (High Priority, High Value)

**Concept:** Run CPU reference in parallel, compare results

```rust
struct ReferenceImpl {
    // Pure Rust implementation of same algorithm
}

impl ReferenceImpl {
    fn split_tet(&mut self, old_tet: usize, point: usize) -> [usize; 4] {
        // CPU implementation (slow but correct)
    }
}

// In test:
let mut gpu_state = GpuState::new(...).await;
let mut cpu_state = ReferenceImpl::new(...);

// Run same operations
gpu_state.split(old_tet, point);
let gpu_result = gpu_state.snapshot().await;

cpu_state.split(old_tet, point);
let cpu_result = cpu_state.snapshot();

assert_eq!(gpu_result.tets, cpu_result.tets, "Tets don't match!");
```

**Pros:**
- Ground truth comparison
- Catches logic bugs
- Can bisect: which operation diverges?

**Cons:**
- Need to write CPU implementation
- Slow for large tests

### 4.2 Differential Testing (Medium Priority, High Value)

**Concept:** Compare working vs broken versions

```rust
// Run same test on both versions
let result_3tet = run_with_3tet_allocation(&points);
let result_4tet = run_with_4tet_allocation(&points);

// Compare intermediate states
compare_snapshots(result_3tet.after_split, result_4tet.after_split);
```

**Pros:**
- Isolates what changed
- Shows minimal difference

### 4.3 Property-Based Testing (Medium Priority, Medium Effort)

**Concept:** Check invariants hold after each operation

```rust
fn check_invariants(state: &GpuState) {
    // Euler characteristic: V - E + F - T = 1 (for convex hull)
    let euler = count_vertices() - count_edges() + count_faces() - count_tets();
    assert_eq!(euler, 1, "Euler violation!");

    // All alive tets have valid vertices
    // All adjacencies have back-pointers
    // No overlapping tets
    // etc.
}

// After every operation:
state.split(...);
check_invariants(&state);
```

## Category 5: Visualization Tools

### 5.1 Memory Access Visualizer (Low Priority, High Value)

**Concept:** Visualize which threads access which memory

```
Thread 0: ████████░░░░░░░░ tets[0-10]
Thread 1: ░░░░░░░░████████ tets[10-20]
Thread 2: ████░░░░████░░░░ tets[0-5, 10-15]  ← OVERLAP!
Thread 3: ░░░░████████░░░░ tets[5-15]  ← CONFLICT with Thread 2!
```

**Implementation:** Log all memory accesses to trace buffer, then visualize

### 5.2 Execution Timeline (Medium Priority, High Value)

**Concept:** Show when each thread executes each operation

```
Time →
Thread 0: [START][ALLOC][WRITE_T0][WRITE_T1]────────────────[COMPLETE]
Thread 1: [START]────[ALLOC]────────[WRITE_T0][CRASH!]
Thread 2: [START][ALLOC][WRITE_T0][WRITE_T1][WRITE_T2]──────[COMPLETE]
Thread 3: ──────[START][ALLOC]──────────────────────────────[COMPLETE]
```

**Implementation:** Timestamp each breadcrumb, visualize

### 5.3 3D Mesh Visualizer (Low Priority, High Value)

**Concept:** Render the tetrahedral mesh to see corruption visually

```rust
// Export to VTK format
fn export_vtk(state: &GpuState, path: &str) {
    // Write .vtk file with tets and adjacency
}

// View in ParaView or custom viewer
// Corrupted tets will be visible (overlapping, inverted, etc.)
```

## Category 6: Existing Tools We Should Use

### 6.1 WGPU Validation Layers (High Priority, ZERO Effort)

**Enable maximum validation:**
```rust
let device = adapter.request_device(&wgpu::DeviceDescriptor {
    required_features: wgpu::Features::empty(),
    required_limits: wgpu::Limits::default(),
    memory_hints: wgpu::MemoryHints::Performance,
}, None).await.unwrap();

// Set environment variables:
// WGPU_VALIDATION=1
// RUST_LOG=wgpu=trace
```

**Pros:**
- Catches many GPU errors
- Free, built-in

**Cons:**
- May not catch all issues
- Overhead in debug builds

### 6.2 RenderDoc (High Priority, Low Effort)

**Use RenderDoc to capture GPU frame:**
1. Run test with RenderDoc attached
2. Capture the problematic dispatch
3. Inspect:
   - Exact shader source with line numbers
   - Buffer contents before/after
   - Pipeline state
   - Execution stats

**Pros:**
- Industry-standard tool
- Deep GPU inspection

**Cons:**
- Compute shader support varies
- Learning curve

### 6.3 Nsight Compute / AMD GPU Profiler (Medium Priority, Medium Effort)

**For NVIDIA GPUs:**
- Nsight Compute can profile compute shaders
- Shows exact line-by-line execution
- Memory access patterns
- Warp divergence

**For AMD GPUs:**
- Radeon GPU Profiler
- Similar features

**Cons:**
- Vendor-specific
- May not support WGPU

## Recommendation: Tiered Approach

### Tier 1: Implement Immediately (1-2 hours)
1. **Breadcrumb Trail** - Shows exactly where crash happens
2. **Per-Thread Debug Slots** - Inspect values from specific thread
3. **Buffer Validator** - Catch corruption early

### Tier 2: High Value (1 day)
4. **Trace Buffer** - General-purpose value logging
5. **Memory Snapshot Tool** - Compare before/after
6. **RenderDoc** - Use existing tool for deep inspection

### Tier 3: Long Term (1 week)
7. **Reference Implementation** - CPU version for ground truth
8. **Execution Timeline Visualizer** - See thread interactions
9. **Property-Based Testing** - Automated invariant checking

## Code Example: Minimal Debugging Framework

```rust
// src/debug.rs
pub struct DebugBuffers {
    pub breadcrumbs: wgpu::Buffer,  // u32 per thread
    pub thread_debug: wgpu::Buffer,  // 16 * vec4<u32> per thread
    pub trace_buffer: wgpu::Buffer,  // Large append-only buffer
    pub trace_index: wgpu::Buffer,   // Atomic counter
}

impl DebugBuffers {
    pub async fn read_breadcrumbs(&self, device: &Device, queue: &Queue, num_threads: usize)
        -> Vec<u32> { ... }

    pub async fn read_thread_debug(&self, device: &Device, queue: &Queue, thread_id: usize)
        -> [Vec4<u32>; 16] { ... }

    pub async fn read_trace(&self, device: &Device, queue: &Queue)
        -> Vec<u32> { ... }

    pub fn print_crashed_threads(&self, breadcrumbs: &[u32]) {
        for (tid, crumb) in breadcrumbs.iter().enumerate() {
            if *crumb != BREADCRUMB_COMPLETE {
                eprintln!("Thread {} stopped at: {:?}", tid,
                    breadcrumb_name(*crumb));
            }
        }
    }
}
```

```wgsl
// src/shaders/debug_lib.wgsl
// Include in all shaders that need debugging

@group(1) @binding(0) var<storage, read_write> breadcrumbs: array<u32>;
@group(1) @binding(1) var<storage, read_write> thread_debug: array<vec4<u32>>;
@group(1) @binding(2) var<storage, read_write> trace_buffer: array<u32>;
@group(1) @binding(3) var<storage, read_write> trace_index: atomic<u32>;

fn breadcrumb(tid: u32, crumb: u32) {
    breadcrumbs[tid] = crumb;
}

fn debug_slot(tid: u32, slot: u32, values: vec4<u32>) {
    thread_debug[tid * 16u + slot] = values;
}

fn trace(value: u32) {
    let idx = atomicAdd(&trace_index, 1u);
    if idx < 1000000u {
        trace_buffer[idx] = value;
    }
}
```

This framework would give us surgical precision in finding bugs!
