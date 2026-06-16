# AI Infra 4 周学习计划

> 基于 2026-06-06 当前 `LearningNotes` 目录结构整理。重点：把 LLM/VLM/VLA 基础与 CUDA、ONNX、TensorRT、vLLM 和 AI 基础设施概念串起来。

## 目标

建立一条可执行的 AI 基础设施学习路径，把现有笔记变成“学习 - 复盘 - 验证”的循环：

- 理解模型能力栈：`LLM -> VLM -> VLA`
- 理解部署工程栈：`PyTorch -> ONNX -> TensorRT / vLLM -> service`
- 建立 GPU 计算、内存带宽、KV Cache、batching、量化、推理瓶颈等工程直觉
- 每周产出一份小结或一个可运行的验证记录

## 资料地图

| 方向 | 现有笔记 | 在计划中的作用 |
|---|---|---|
| LLM 基础 | `LLM/notes.md` | Transformer、tokenizer、RoPE、KV Cache、attention、模型内部机制 |
| 多模态与机器人 | `VLM/notes.md`, `VLA/notes.md`, `ROS2/notes.md` | 从语言/视觉模型过渡到动作决策与机器人部署 |
| AI 基础设施 | `Infra/AIInfra-DEEPSEEK.md` | GPU vs CPU、FLOPS、带宽、屋顶线模型、分布式训练 |
| GPU 编程 | `Cuda/notes.md` | Thread/block/grid、内存层级、tiled matmul、PyTorch CUDA 测时 |
| 模型导出 | `ONNX/notes.md` | ONNX 作为 IR、opset、计算图结构、导出与兼容性检查 |
| 推理优化 | `TensorRT/notes.md`, `vLLM/notes.md` | TensorRT engine 流程、FP16/INT8、PagedAttention、continuous batching |

## 周计划

| 周次 | 日期 | 核心问题 | 主要笔记 | 交付物 |
|---|---:|---|---|---|
| 第 1 周 | 2026-06-08 至 2026-06-14 | Transformer 如何变成一个推理负载？ | `LLM/notes.md`, `Infra/AIInfra-DEEPSEEK.md` | 1 页小结：tokenization、attention、RoPE、KV Cache、计算瓶颈 vs 带宽瓶颈 |
| 第 2 周 | 2026-06-15 至 2026-06-21 | 为什么 GPU 架构对 AI 负载很重要？ | `Cuda/notes.md`, `Infra/AIInfra-DEEPSEEK.md` | 小型 benchmark 记录：CPU/GPU matmul 测时、`torch.cuda.synchronize()`、屋顶线直觉 |
| 第 3 周 | 2026-06-22 至 2026-06-28 | 训练好的模型如何变成可部署产物？ | `ONNX/notes.md`, `TensorRT/notes.md` | 部署检查清单：导出、shape/opset 检查、数值对齐、TensorRT profile |
| 第 4 周 | 2026-06-29 至 2026-07-05 | LLM/VLM 推理服务如何变得高效？ | `vLLM/notes.md`, `VLM/notes.md`, `VLA/notes.md` | 最终知识地图：vLLM batching/KV Cache、VLM/VLA serving 约束、下一步项目选择 |

## 每日节奏

尽量使用一个 90 分钟学习块：

| 环节 | 时间 | 动作 |
|---|---:|---|
| 回忆 | 10 分钟 | 阅读前先写下自己还记得什么。 |
| 阅读 | 35 分钟 | 学习目标笔记中的一个聚焦章节。 |
| 重建 | 25 分钟 | 不看原文，重新画出图、表格或代码流程。 |
| 验证 | 15 分钟 | 跑一个小例子、检查一条命令，或口头解释一个瓶颈。 |
| 记录 | 5 分钟 | 写 3 条 bullet：学到了什么、不清楚什么、下一步做什么。 |

## 第 1 周细化：LLM 推理负载基础

重点章节：

- `LLM/notes.md`：tokenizer、embedding、位置编码、RoPE
- `Infra/AIInfra-DEEPSEEK.md`：GPU/CPU 差异、FLOPS、内存带宽、屋顶线模型

检查点：

- 用一段话解释 encoder-decoder 与 decoder-only 的区别。
- 画出从文本输入到 token IDs、embedding、logits 的路径。
- 解释为什么 KV Cache 会随序列长度增长。
- 尝试判断 attention、matmul、LayerNorm、sampling 更偏计算瓶颈还是带宽瓶颈。

## 第 2 周细化：GPU 与 CUDA 直觉

重点章节：

- `Cuda/notes.md`：thread/block/grid、内存层级、reduction、tiled matmul
- `Infra/AIInfra-DEEPSEEK.md`：DP/TP/PP 与通信瓶颈

检查点：

- 解释 `threadIdx`、`blockIdx`、`blockDim`、`gridDim`。
- 解释 coalesced access、shared memory 和 bank conflict。
- 运行或改写 PyTorch 测时示例，并包含 `torch.cuda.synchronize()`。
- 写一小段说明：为什么 matmul 比复杂分支逻辑更适合 GPU。

## 第 3 周细化：ONNX 到 TensorRT 部署

重点章节：

- `ONNX/notes.md`：计算图结构、opset、dynamic axes、部署链路
- `TensorRT/notes.md`：engine 构建流程、FP16/INT8、`trtexec`、dynamic shape profiles

检查点：

- 解释为什么 ONNX 是中间表示，而不是性能保证。
- 创建一份模型导出正确性检查清单。
- 解释 FP32、FP16、INT8 的取舍。
- 总结构建和 profile 一个 TensorRT engine 所需的最小命令集合。

## 第 4 周细化：vLLM、VLM 与 VLA Serving

重点章节：

- `vLLM/notes.md`：PagedAttention、continuous batching、tensor parallel、prefix caching
- `VLM/notes.md`：视觉语言桥接
- `VLA/notes.md`：action head、数据集、ROS2 集成

检查点：

- 解释为什么 KV Cache 内存管理是 vLLM 的核心。
- 对比静态 KV 分配与 PagedAttention。
- 解释 VLM serving 和 LLM serving 有什么不同。
- 解释 VLA 额外带来的约束：动作频率、机器人状态、安全性、ROS2 集成。

## 复盘问题

每周结束时回答：

1. 现在我能脱离笔记解释的一个概念是什么？
2. 哪个概念仍然像是在背，而不是真的理解？
3. 哪个小型可运行验证能让这个概念变具体？
4. 哪篇笔记应该补一个更清晰的图、表格或示例？

## 完成标准

到 2026-07-05，如果存在以下产物，就算完成本计划：

- 每周一份学习小结
- 一份 GPU 或 PyTorch 测时记录
- 一份 ONNX/TensorRT 部署检查清单
- 一份最终 AI infra 知识地图，串联 LLM/VLM/VLA 与部署、推理服务

