# TensorRT

> NVIDIA TensorRT 高性能推理引擎学习笔记

## 内容概览

- TensorRT 工作流程（网络定义 → 构建 Engine → 推理）
- 精度优化（FP32 → FP16 → INT8）
- 从 ONNX 构建 TensorRT Engine
- Python API 与 C++ API
- 性能分析与调优

## 1. TensorRT 全流程

```
PyTorch/TensorFlow 模型
      │
      ▼
ONNX 模型（torch.onnx.export）
      │
      ▼
TensorRT Engine 构建（trtexec / Python API）
 ├── 解析 ONNX 计算图
 ├── 图优化（算子融合、常量折叠）
 ├── 精度校准（INT8 需校准数据集）
 └── 生成 .engine / .trt 文件（特定 GPU 绑定）
      │
      ▼
TensorRT Runtime 推理
 ├── 分配 GPU 输入/输出 buffer
 ├── context.execute_async_v3(...)
 └── 读取输出结果
```

**核心优势**：
- 自动算子融合（Conv+BN+ReLU → 单个 CUDA Kernel）
- Tensor Core 利用（FP16/INT8 在 Tensor Core 上极快）
- 动态 Shape 支持（显式设置 min/opt/max profile）
- 延迟通常比 PyTorch 推理低 2-10 倍

---

## 2. 精度模式对比

| 精度 | 内存占用 | 速度 | 精度损失 | 说明 |
|------|---------|------|---------|------|
| FP32 | 100% | 基准 | 无 | 与训练精度一致 |
| **FP16** | ~50% | 2-3x | 极小 | 推荐首选，Tensor Core |
| **INT8** | ~25% | 3-5x | 较小 | 需校准数据集（PTQ）|
| INT4 | ~12% | 更快 | 较大 | LLM 量化 |

**FP16 是生产部署首选**：基本无精度损失，速度显著提升。

---

## 3. 关键 API 概念

| 对象 | 说明 |
|------|------|
| `IBuilder` | 构建 TensorRT Engine 的入口 |
| `INetworkDefinition` | 网络计算图定义 |
| `IBuilderConfig` | 配置精度、优化 profile 等 |
| `ICudaEngine` | 构建好的推理引擎（序列化后可保存） |
| `IExecutionContext` | 推理执行上下文（可复用） |
| `trtexec` | 命令行工具，快速转换和性能测试 |

---

## 4. trtexec 命令行工具

```bash
# FP32 转换
trtexec --onnx=model.onnx --saveEngine=model_fp32.trt

# FP16 转换（推荐）
trtexec --onnx=model.onnx --fp16 --saveEngine=model_fp16.trt

# INT8（适合 Q/DQ 量化模型，或已有 calibration cache 的场景）
trtexec --onnx=model.onnx --int8 --saveEngine=model_int8.trt

# 使用已有 INT8 calibration cache
trtexec --onnx=model.onnx --int8 --calib=calibration.cache \
  --saveEngine=model_int8.trt

# 动态 Shape（指定 min、opt、max）
trtexec --onnx=model.onnx --fp16 \
  --minShapes=input:1x3x224x224 \
  --optShapes=input:8x3x224x224 \
  --maxShapes=input:32x3x224x224 \
  --saveEngine=model_dynamic.trt

# 性能测试
trtexec --loadEngine=model_fp16.trt --iterations=1000 --warmUp=200
```

常用调试参数：

```bash
# 打印网络层信息
trtexec --onnx=model.onnx --fp16 --dumpLayerInfo --profilingVerbosity=detailed

# 导出 profile，便于定位耗时层
trtexec --loadEngine=model_fp16.trt --dumpProfile --exportProfile=profile.json

# 检查解析失败原因
trtexec --onnx=model.onnx --verbose
```

注意：`trtexec --int8` 可以快速测试 INT8 构建能力，但真实项目里更推荐使用校准器、Q/DQ 量化模型或框架导出的显式量化模型来控制量化精度。

---

## 5. INT8 校准（PTQ）

```python
class MyCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, data_loader, cache_file):
        super().__init__()
        self.data_loader = iter(data_loader)
        self.cache_file = cache_file
        self.device_input = None  # 在 GPU 上的 buffer

    def get_batch_size(self):
        return BATCH_SIZE

    def get_batch(self, names):
        try:
            batch = next(self.data_loader).numpy()
            cuda.memcpy_htod(self.device_input, batch)
            return [int(self.device_input)]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
```

---

## 6. Triton 推理服务器（生产部署）

TensorRT Engine 可直接部署到 **NVIDIA Triton**，支持：
- 多模型并发服务
- 动态 batch（自动合并多个请求）
- gRPC / HTTP 接口
- 性能监控指标

```bash
# 启动 Triton（Docker）
docker run --gpus all -p 8000:8000 -p 8001:8001 \
  -v /models:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

---

## 7. TensorRT 核心概念补充

### 7.1 Engine 不是通用模型文件

TensorRT 生成的 `.engine` / `.plan` 是**针对具体环境优化后的二进制推理计划**，通常和以下因素强相关：

| 因素 | 影响 |
|---|---|
| GPU 架构 | 不同 GPU 的 Tensor Core、SM、内存层次不同 |
| TensorRT 版本 | builder 和 tactic 选择可能变化 |
| CUDA / cuDNN / driver | 影响 kernel 和 runtime 兼容 |
| 输入 shape / profile | 动态 shape 下只对 profile 范围内有效 |
| 精度模式 | FP32、TF32、FP16、INT8 的 tactic 不同 |

因此工程上常见做法是：

- ONNX 可以跨机器保存和分发。
- TensorRT engine 最好在目标部署环境构建，或者至少在同型号 GPU、同软件栈环境构建。
- engine 需要和模型版本、TensorRT 版本、GPU 型号一起管理。

### 7.2 Builder、Engine、Context 的关系

```text
Builder + Network + BuilderConfig
        |
        | build_serialized_network
        v
Serialized Engine (.plan/.engine)
        |
        | Runtime 反序列化
        v
ICudaEngine
        |
        | create_execution_context
        v
IExecutionContext
        |
        | enqueue / execute_async
        v
GPU 推理
```

| 对象 | 生命周期 | 重点 |
|---|---|---|
| Builder | 离线构建阶段 | 慢，负责搜索 tactic 和生成 engine |
| Engine | 部署阶段可复用 | 保存网络结构、kernel 选择、binding 信息 |
| ExecutionContext | 推理阶段 | 保存动态 shape、运行状态，可为并发创建多个 context |

面试常问点：**engine 构建慢是正常的**，因为 TensorRT 会为每层尝试多种 kernel/tactic，选择目标 GPU 上更快的实现。

## 8. 动态 Shape 和 Optimization Profile

TensorRT 的动态 shape 不是“随便什么 shape 都能跑”，而是需要在构建时声明每个动态输入的范围：

```bash
trtexec --onnx=model.onnx --fp16 \
  --minShapes=images:1x3x224x224 \
  --optShapes=images:8x3x224x224 \
  --maxShapes=images:32x3x224x224 \
  --saveEngine=model_dynamic.engine
```

| Profile 参数 | 含义 | 选择建议 |
|---|---|---|
| min | 支持的最小 shape | 业务真实最小值 |
| opt | 最常见、最希望优化的 shape | 线上主流 batch/尺寸 |
| max | 支持的最大 shape | 不要无脑设太大，会增加构建和内存压力 |

推理时必须设置输入 shape：

```python
context.set_input_shape("images", np_input.shape)
```

常见坑：

- `maxShapes` 设得过大，engine 体积和显存占用变高，性能反而下降。
- `optShapes` 不是线上常见输入，导致真实性能不理想。
- 多输入模型的 profile 必须同时覆盖所有输入。
- 动态 H/W 比动态 batch 更复杂，优先固定 H/W，只动态 batch。

## 9. 精度模式与量化

### 9.1 FP32、TF32、FP16、INT8

| 精度 | 说明 | 工程建议 |
|---|---|---|
| FP32 | 标准单精度 | 作为精度基线 |
| TF32 | Ampere 及以后 GPU 对 FP32 矩阵计算的加速格式 | 默认可能启用，精度接近 FP32 |
| FP16 | 半精度，Tensor Core 友好 | 首选加速方案 |
| INT8 | 8 bit 整数量化 | 需要校准/QAT/QDQ，适合追求吞吐 |

一般项目顺序：

```text
先跑 FP32 对齐
  -> 开 FP16，看精度是否可接受
  -> 再尝试 INT8，重点处理校准和敏感层
```

### 9.2 INT8 的两种常见路线

| 路线 | 说明 | 特点 |
|---|---|---|
| PTQ 校准 | TensorRT builder 用校准数据统计激活范围 | 不改训练流程，成本低 |
| 显式量化 Q/DQ | ONNX 图里已有 Quantize/DeQuantize 节点 | 更可控，常用于 QAT 或导出量化模型 |

INT8 不是“打开 `--int8` 就一定更快更准”。需要关注：

- 校准数据是否覆盖真实分布。
- 是否有层对量化敏感。
- 部分 op 是否没有 INT8 kernel，导致反复 reformat 或 fallback。
- 输出后处理是否对小数误差敏感，例如检测框、分割边界。

## 10. TensorRT 为什么快

| 优化 | 解释 |
|---|---|
| Layer fusion | Conv + Bias + BN + ReLU 等融合，减少 kernel launch 和显存读写 |
| Tactic search | 为每层选择更快的 kernel 实现 |
| Precision lowering | FP16/INT8 使用 Tensor Core |
| Memory reuse | 复用中间 tensor buffer，降低显存占用 |
| Kernel auto-tuning | 根据 GPU、shape、workspace 搜索最优实现 |
| Dynamic batching | 配合 Triton 可以合并请求，提高吞吐 |

一个重要判断：**TensorRT 优化的是模型主体计算，不自动优化所有业务前后处理**。如果 resize、decode、NMS、tokenization、后处理仍在 CPU 上，端到端延迟可能不会明显下降。

## 11. 性能测试方法

### 11.1 正确 benchmark 要分层

```text
端到端延迟
  = 数据读取
  + 前处理
  + H2D 拷贝
  + TensorRT enqueue
  + D2H 拷贝
  + 后处理
```

建议分别记录：

| 指标 | 含义 |
|---|---|
| latency | 单请求耗时，关注 P50/P95/P99 |
| throughput | 每秒处理样本数，关注 batch 和并发 |
| GPU utilization | GPU 是否吃满 |
| H2D/D2H time | CPU-GPU 拷贝是否成为瓶颈 |
| layer profile | 哪些层耗时最高 |

### 11.2 warmup 很重要

第一次推理往往包含 lazy init、cache、显存分配等开销。benchmark 前要 warmup：

```bash
trtexec --loadEngine=model.engine --warmUp=500 --duration=10
```

### 11.3 CUDA Graph

如果输入 shape 固定、推理流程稳定，可以尝试 CUDA Graph 减少 CPU enqueue 开销。它对小模型、低延迟服务更明显；大模型通常瓶颈在 GPU 计算本身。

## 12. 常见部署坑

| 问题 | 现象 | 解决思路 |
|---|---|---|
| ONNX 解析失败 | parser 报 unsupported op | 降 opset、改写算子、写 plugin |
| engine 换机器不能用 | deserialize 失败或结果异常 | 在目标 GPU/版本重建 engine |
| 动态 shape 报错 | shape 超出 profile | 检查 min/opt/max 和推理输入 |
| FP16 精度异常 | 输出偏差明显 | 找敏感层，局部保持 FP32 或检查输入归一化 |
| INT8 掉点 | mAP/acc 降很多 | 改校准集、QDQ/QAT、排除敏感层 |
| 性能没提升 | GPU 利用率低 | 检查 batch、数据拷贝、CPU 前后处理、fallback |
| 显存暴涨 | build 或 runtime OOM | 降 max shape、限制 workspace、减小 batch |
| NMS 很慢 | 主体快但后处理慢 | 使用 EfficientNMS plugin 或 GPU 后处理 |

## 13. TensorRT Plugin

当 ONNX 里有 TensorRT 不支持的算子，或者某段后处理想放到 GPU 上，可以写 plugin。

常见场景：

- YOLO/DETR 类模型的 NMS 后处理。
- 自定义激活函数或特殊算子。
- 模型里有 TensorRT parser 不支持的 op。
- 想融合多个小 op，减少 kernel launch。

工程建议：

1. 能用标准 ONNX 算子表达就不要写 plugin。
2. 能通过 TensorRT 官方 plugin 解决就优先用官方 plugin。
3. 自定义 plugin 要考虑序列化、版本兼容、动态 shape、精度模式和测试成本。

## 14. 与 ONNX Runtime / Triton 的关系

| 工具 | 角色 | 典型用法 |
|---|---|---|
| TensorRT | NVIDIA GPU 上的高性能推理引擎 | 构建 engine，嵌入 C++/Python 服务 |
| ONNX Runtime TensorRT EP | ORT 调 TensorRT 后端 | 快速接入，但可控性略弱 |
| Triton Inference Server | 推理服务框架 | 多模型、多实例、动态 batch、HTTP/gRPC |

选择建议：

- 单模型、本地嵌入式服务：直接 TensorRT Runtime。
- 已有 ORT 推理链路：尝试 TensorRT Execution Provider。
- 多模型线上服务：Triton + TensorRT engine。

## 15. 面试速记

**Q：TensorRT 是什么？**

TensorRT 是 NVIDIA 面向推理部署的高性能优化器和 runtime。它可以从 ONNX 等模型构建 engine，通过算子融合、tactic search、FP16/INT8、内存复用等方式降低延迟、提升吞吐。

**Q：engine 为什么不能随便跨机器拷贝？**

因为 engine 是针对 GPU 架构、TensorRT/CUDA/driver 版本、输入 profile 和精度模式优化出来的二进制计划，不是通用模型格式。跨环境部署建议重新 build。

**Q：动态 shape 怎么做？**

构建时为每个动态输入设置 optimization profile，包括 min/opt/max shape；推理时用 `context.set_input_shape` 设置本次输入。`opt` 应该贴近线上最常见 shape。

**Q：FP16 和 INT8 怎么选？**

先用 FP32 做精度基线，再开 FP16，通常收益明显且精度损失小；INT8 需要校准或 QAT/QDQ，收益更高但精度风险也更高。

**Q：TensorRT 性能不如预期怎么办？**

先分解端到端耗时，区分前处理、H2D、enqueue、D2H、后处理；再用 `trtexec --dumpProfile` 看层级耗时，检查是否有 unsupported op、reformat、fallback、batch 太小或 CPU 后处理瓶颈。

参考资料：

- NVIDIA TensorRT Developer Guide：<https://docs.nvidia.com/deeplearning/tensorrt/latest/>
- TensorRT `trtexec` 工具：<https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html>
- TensorRT Dynamic Shapes：<https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-dynamic-shapes.html>
- TensorRT Performance Best Practices：<https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html>
- TensorRT Accuracy Considerations：<https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html>

```python
"""
TensorRT Python API 完整工作流
需要：pip install tensorrt pycuda
环境：NVIDIA GPU + CUDA + TensorRT 安装包
"""
import os
import numpy as np
import torch
import torch.nn as nn

# ===== Step 1：准备 ONNX 模型 =====
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(c)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + x)

class SmallResNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(ResBlock(64), ResBlock(64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(64, num_classes))
    def forward(self, x):
        return self.head(self.body(self.stem(x)))

model = SmallResNet().eval()
dummy = torch.randn(1, 3, 224, 224)
onnx_path = "resnet_small.onnx"

torch.onnx.export(model, dummy, onnx_path,
                  opset_version=17,
                  input_names=["images"],
                  output_names=["logits"],
                  dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}})
print(f"ONNX 导出完成: {onnx_path} ({os.path.getsize(onnx_path)/1024:.1f} KB)")

# ===== Step 2：TensorRT 构建 Engine =====
try:
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    def build_engine(onnx_file, fp16=True, max_batch=16):
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        config = builder.create_builder_config()

        # 配置精度
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  已启用 FP16 Tensor Core 优化")

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        # 解析 ONNX
        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_file, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("ONNX 解析失败")

        # 设置动态 shape profile
        profile = builder.create_optimization_profile()
        profile.set_shape("images",
                          min=(1,  3, 224, 224),
                          opt=(8,  3, 224, 224),
                          max=(max_batch, 3, 224, 224))
        config.add_optimization_profile(profile)

        print("  正在构建 Engine（可能需要几分钟）...")
        engine = builder.build_serialized_network(network, config)
        return engine

    # 构建并保存 engine
    engine_bytes = build_engine(onnx_path, fp16=True)
    engine_path = "resnet_small_fp16.trt"
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    print(f"Engine 保存: {engine_path} ({os.path.getsize(engine_path)/1024/1024:.1f} MB)")

    # ===== Step 3：TensorRT 推理 =====
    import pycuda.driver as cuda
    import pycuda.autoinit

    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    def trt_inference(context, engine, np_input, batch_size=4):
        context.set_input_shape("images", np_input.shape)

        # 分配 GPU 内存
        d_input  = cuda.mem_alloc(np_input.nbytes)
        output_shape = (batch_size, 100)
        d_output = cuda.mem_alloc(np.empty(output_shape, dtype=np.float32).nbytes)

        # H2D → 推理 → D2H
        cuda.memcpy_htod(d_input, np_input)
        context.execute_v2([int(d_input), int(d_output)])
        output = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output, d_output)
        return output

    test_input = np.random.randn(4, 3, 224, 224).astype(np.float32)
    result = trt_inference(context, engine, test_input, batch_size=4)
    print(f"\nTRT 推理输出形状: {result.shape}")
    print(f"预测类别: {result.argmax(axis=1)}")

    # ===== Step 4：性能对比 =====
    import time

    def benchmark_trt(n=100):
        inp = np.random.randn(8, 3, 224, 224).astype(np.float32)
        for _ in range(10): trt_inference(context, engine, inp, 8)
        t0 = time.perf_counter()
        for _ in range(n): trt_inference(context, engine, inp, 8)
        return (time.perf_counter() - t0) / n * 1000

    def benchmark_pt(n=100):
        inp = torch.randn(8, 3, 224, 224)
        with torch.no_grad():
            for _ in range(10): model(inp)
            t0 = time.perf_counter()
            for _ in range(n): model(inp)
        return (time.perf_counter() - t0) / n * 1000

    t_trt = benchmark_trt()
    t_pt = benchmark_pt()
    print(f"\n[Batch=8 性能对比]")
    print(f"PyTorch CPU: {t_pt:.2f} ms")
    print(f"TensorRT:    {t_trt:.2f} ms")
    print(f"加速比: {t_pt/t_trt:.1f}x")

    # 清理
    for f in [onnx_path, engine_path]:
        if os.path.exists(f): os.remove(f)

except ImportError:
    print("\nTensorRT 未安装。安装方式：")
    print("1. 从 NVIDIA 官网下载 TensorRT tar 包")
    print("   https://developer.nvidia.com/tensorrt")
    print("2. 安装 Python 绑定: pip install tensorrt")
    print("3. 或使用 Docker: nvcr.io/nvidia/tensorrt:24.01-py3")
    print("\n命令行工具 trtexec 是最快的测试方式：")
    print("  trtexec --onnx=model.onnx --fp16 --saveEngine=model.trt")
    # 清理
    if os.path.exists(onnx_path): os.remove(onnx_path)
```
