# CUDA to WGSL Translation Quick Reference

## Quick Substitution Patterns

### Thread/Block Indexing
```cuda
threadIdx.x/y/z          →  local_invocation_id.x/y/z
blockIdx.x/y/z           →  workgroup_id.x/y/z
blockDim.x/y/z           →  (compile-time constant in @workgroup_size)
gridDim.x/y/z            →  num_workgroups (via push constants)
```

### Memory Spaces
```cuda
__shared__ int data[64]  →  var<workgroup> data: array<i32, 64>
__device__ int* ptr      →  var<storage, read_write> (via bindings)
__constant__ int val     →  var<uniform> or const
```

### Synchronization
```cuda
__syncthreads()          →  workgroupBarrier()
```

### Atomics (direct equivalents!)
```cuda
atomicAdd(&x, val)       →  atomicAdd(&x, val)
atomicSub(&x, val)       →  atomicSub(&x, val)
atomicMin(&x, val)       →  atomicMin(&x, val)
atomicMax(&x, val)       →  atomicMax(&x, val)
atomicExch(&x, val)      →  atomicExchange(&x, val)
```

### Data Types
```cuda
int                      →  i32
unsigned int / uint      →  u32
float                    →  f32
int2/3/4                 →  vec2/3/4<i32>
uint2/3/4                →  vec2/3/4<u32>
float2/3/4               →  vec2/3/4<f32>
```

### Control Flow (Same!)
```cuda
if/else/for/while        →  if/else/for/while (same syntax!)
```

### Functions
```cuda
__device__ int foo()     →  fn foo() -> i32
__global__ void kernel() →  @compute @workgroup_size(64) fn kernel()
```

## Translation Workflow

1. **Copy the CUDA kernel**
2. **Replace patterns** using table above
3. **Convert array accesses** to buffer bindings
4. **Add proper WGSL syntax** (let, var, return types)
5. **Test incrementally**

## Example Translation

### CUDA Version:
```cuda
__global__ void myKernel(int* data, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    __shared__ int temp[256];
    temp[threadIdx.x] = data[idx];
    __syncthreads();

    data[idx] = temp[threadIdx.x] + 1;
}
```

### WGSL Version:
```wgsl
@group(0) @binding(0) var<storage, read_write> data: array<i32>;

@compute @workgroup_size(256)
fn my_kernel(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let idx = gid.x;
    let count = arrayLength(&data);
    if idx >= count { return; }

    var<workgroup> temp: array<i32, 256>;
    temp[lid.x] = data[idx];
    workgroupBarrier();

    data[idx] = temp[lid.x] + 1;
}
```

## Now You Try!

Open `gDel3D/GDelFlipping/src/gDel3D/GPU/KerDivision.cu` and start translating:

1. Start with helper functions (lines 61-82) - simple encoding
2. Then kerMakeFirstTetra (lines 46-95) - fixed initialization
3. Then kerSplitTetra (lines 97-192) - the big one!

Use the patterns above and it should be fairly mechanical!
