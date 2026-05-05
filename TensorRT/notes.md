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

# INT8（需要校准数据）
trtexec --onnx=model.onnx --int8 --calib=calib_data/ --saveEngine=model_int8.trt

# 动态 Shape（指定 min、opt、max）
trtexec --onnx=model.onnx --fp16 \
  --minShapes=input:1x3x224x224 \
  --optShapes=input:8x3x224x224 \
  --maxShapes=input:32x3x224x224 \
  --saveEngine=model_dynamic.trt

# 性能测试
trtexec --loadEngine=model_fp16.trt --batch=8 --iterations=1000
```

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
