# CUDA 并行计算

> NVIDIA CUDA 编程学习笔记

## 内容概览

- CUDA 线程模型（Thread、Block、Grid）
- 内存层级（Global、Shared、Register）
- Kernel 编写与启动
- 并行归约、矩阵乘法优化
- 与 PyTorch 的交互（Custom CUDA Extension）

## 1. 线程模型：Thread → Block → Grid

GPU 并行的本质：同一段代码（Kernel）由数以千计的线程并发执行。

### 三级层次结构

```
Grid（整个任务）
  └── Block（线程块，共享 Shared Memory，可同步）
        └── Thread（最小执行单元）
```

| 层级 | 索引变量 | 说明 |
|------|---------|------|
| Thread | `threadIdx.x/y/z` | 线程在 Block 内的位置 |
| Block | `blockIdx.x/y/z` | Block 在 Grid 内的位置 |
| 维度信息 | `blockDim`, `gridDim` | Block/Grid 的尺寸 |

### 全局线程索引（1D 示例）

$$\text{global\_idx} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$

### C++/CUDA Kernel 语法

```cpp
// __global__ 表示在 GPU 上运行，从 CPU 调用
__global__ void vector_add(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {          // 边界检查：防止越界
        C[idx] = A[idx] + B[idx];
    }
}

// 启动 Kernel：<<<gridDim, blockDim>>>
int N = 1024 * 1024;
int threads = 256;
int blocks  = (N + threads - 1) / threads;   // 向上取整
vector_add<<<blocks, threads>>>(d_A, d_B, d_C, N);
cudaDeviceSynchronize();   // 等待 GPU 完成
```

---

## 2. 内存层级

| 类型 | 速度 | 生命周期 | 容量 | 关键字 |
|------|------|---------|------|--------|
| **Register** | 最快 | 单线程 | ~256KB/SM | 自动 |
| **Shared Memory** | ~L1 Cache | 同一 Block | ~48KB/Block | `__shared__` |
| **L2 Cache** | 中 | 自动 | 数 MB | 自动 |
| **Global Memory（显存）** | 慢（~600 GB/s） | 全程序 | GB 级 | `cudaMalloc` |
| **Constant Memory** | 快（有广播缓存） | 只读 | 64KB | `__constant__` |

**核心优化原则**：
- 访存合并（Coalesced Access）：同一 Warp 的 32 个线程访问连续地址 → 合并为单次事务
- 用 Shared Memory 做 Tile 缓存，减少 Global Memory 访问次数
- 避免 Bank Conflict：Shared Memory 有 32 个 Bank，多线程同时访同一 Bank 会串行化

---

## 3. 并行归约（Parallel Reduction）

将 N 个元素聚合为 1 个值（如求和），利用 log₂N 轮次完成：

```
Round 1: [1,2,3,4,5,6,7,8] → [3,7,11,15]   (相邻两两相加)
Round 2: [3,7,11,15]       → [10,26]
Round 3: [10,26]           → [36]
```

**关键**：使用 Shared Memory 存中间结果 + `__syncthreads()` 同步。

---

## 4. Tiled 矩阵乘法

将 A、B 矩阵分成 TILE×TILE 的小块，每个 Block 负责计算 C 的一块：

```
for k_tile in range(K / TILE_SIZE):
    # 1. 协同将 A_tile、B_tile 载入 Shared Memory
    # 2. __syncthreads() 等待所有线程完成载入
    # 3. 计算局部点积，累加到寄存器
    # 4. __syncthreads() 再进行下一轮
```

Global Memory 访问量减少约 TILE_SIZE 倍，典型 TILE=32 → 32 倍带宽节省。

---

## 5. 与 PyTorch 的交互

```
方式一：tensor.cuda() / .to('cuda')    ← 数据移到 GPU
方式二：torch.cuda.synchronize()       ← 同步 GPU 操作（测时必须）
方式三：Custom CUDA Extension          ← 自定义算子，最灵活
方式四：Triton（Python DSL）           ← 比 CUDA C 更易写的 GPU 内核
```

```python
import torch
import time

# ===== 环境检查 =====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")
if device == 'cuda':
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"显存总量: {props.total_memory / 1e9:.1f} GB")
    print(f"SM 数量: {props.multi_processor_count}")
    print(f"CUDA 版本: {torch.version.cuda}")

# ===== 1. 向量运算：CPU vs GPU 速度对比 =====
N = 4096
A_cpu = torch.randn(N, N)
B_cpu = torch.randn(N, N)

def bench(fn, *args, warmup=5, repeat=20):
    for _ in range(warmup):
        fn(*args)
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn(*args)
    if device == 'cuda':
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1000  # ms

t_cpu = bench(torch.matmul, A_cpu, B_cpu)
print(f"\n[{N}x{N} 矩阵乘法]")
print(f"CPU 耗时: {t_cpu:.2f} ms")

if device == 'cuda':
    A_gpu = A_cpu.cuda()
    B_gpu = B_cpu.cuda()
    t_gpu = bench(torch.matmul, A_gpu, B_gpu)
    print(f"GPU 耗时: {t_gpu:.2f} ms")
    print(f"加速比: {t_cpu / t_gpu:.1f}x")

# ===== 2. 内存管理 =====
if device == 'cuda':
    print(f"\n[显存使用]")
    print(f"已分配: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"已缓存: {torch.cuda.memory_reserved() / 1e6:.1f} MB")

    # 释放缓存
    del A_gpu, B_gpu
    torch.cuda.empty_cache()
    print(f"清理后已分配: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

# ===== 3. Numba CUDA：直接写 GPU Kernel（需安装 numba + CUDA）=====
try:
    from numba import cuda
    import numpy as np
    import math

    @cuda.jit
    def gpu_vector_add(A, B, C):
        """向量加法 Kernel"""
        idx = cuda.grid(1)          # 等价于 blockIdx.x * blockDim.x + threadIdx.x
        if idx < A.shape[0]:
            C[idx] = A[idx] + B[idx]

    n = 1 << 20  # 1M 元素
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)

    threads_per_block = 256
    blocks = math.ceil(n / threads_per_block)

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.to_device(c)

    gpu_vector_add[blocks, threads_per_block](d_a, d_b, d_c)
    c_result = d_c.copy_to_host()

    print(f"\n[Numba CUDA 向量加法]")
    print(f"最大误差: {np.max(np.abs(c_result - (a + b))):.2e}")

except ImportError:
    print("\n[Numba 未安装] 安装: pip install numba")
    print("  安装后可直接用 @cuda.jit 编写 CUDA Kernel")

# ===== 4. 自定义 CUDA Extension 流程说明 =====
print("\n===== Custom CUDA Extension 步骤 =====")
print("""
1. 编写 my_op.cu（CUDA Kernel）和 my_op.cpp（PyTorch binding）
   // my_op.cu
   __global__ void my_kernel(float* x, float* y, int n) { ... }
   
   // my_op.cpp
   #include <torch/extension.h>
   torch::Tensor my_forward(torch::Tensor x) { ... }
   PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &my_forward); }

2. 编译加载（JIT 方式）：
   from torch.utils.cpp_extension import load
   my_op = load('my_op', sources=['my_op.cpp', 'my_op.cu'], verbose=True)

3. 调用：
   output = my_op.forward(input_tensor)
""")
```
