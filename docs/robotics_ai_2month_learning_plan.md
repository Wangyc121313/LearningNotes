# 机器人/智驾方向两个月学习计划

> 目标：在已有嵌入式 Linux、C/C++、Linux 系统、音视频、LLM 基础上，用两个月补齐机器人/智驾方向更核心的 AI 技术栈。重点不是把每门课都学完，而是形成一条可展示的能力链：数据与模型训练 -> GPU/推理优化 -> ROS2 系统集成 -> 多模态/VLA/LLM 服务理解。

## 1. 阶段定位

### 1.1 你第一阶段已经有的优势

- C/C++、Linux、进程线程、网络、IO 多路复用、系统调用、V4L2/RTSP/FFmpeg/MPP 等，适合承接机器人/智驾里的端侧工程、传感器接入、数据流处理、部署优化任务。
- LLM 已经有 Transformer、Attention、MHA/GQA/MoE、LoRA、KV Cache、RAG、Agent 等基础，这能帮助你更快理解 VLA、具身智能、多模态策略模型和 LLM 推理服务。
- ONNX/RKNN 虽然当前进度为 0，但和你已有的嵌入式部署方向高度相关，是第二阶段最应该补起来的桥。

### 1.2 第二阶段的主线

两个月内建议围绕四条能力主线推进：

1. 训练能力：PyTorch + 机器学习 + 深度学习
2. 加速能力：CUDA + TensorRT + ONNX/量化/Profiling
3. 机器人系统能力：ROS/ROS2 + 传感器数据流 + 节点通信
4. 前沿方向理解：强化学习 + VLA + vLLM/SGLang 服务化

### 1.3 优先级

| 优先级 | 技术栈 | 原因 |
| --- | --- | --- |
| P0 | PyTorch、深度学习、机器学习、ONNX/TensorRT | 机器人/智驾感知与部署的共同基础 |
| P0 | ROS2 | 机器人岗位高频基础设施，能把模型和传感器接起来 |
| P1 | CUDA | 不一定一开始写复杂 kernel，但必须懂 GPU 执行、显存、stream、profiling |
| P1 | 强化学习、VLA | 面向具身智能/机器人策略，但两个月内以理解框架和复现实验为主 |
| P1 | vLLM、SGLang | 和你的 LLM 基础衔接，理解推理服务、KV cache、调度、吞吐/延迟 |
| P2 | 计算机网络、操作系统、C++、Git | 作为工程底座持续补强，不作为第二阶段主任务 |

## 2. 每个技术栈的具体学习内容

## 2.1 机器学习

### 必学概念

- 监督学习、无监督学习、半监督学习、自监督学习的任务区别。
- 分类、回归、检测、分割、序列预测、策略学习分别解决什么问题。
- 训练集、验证集、测试集、数据泄漏、过拟合、欠拟合。
- 损失函数：MSE、MAE、Cross Entropy、Focal Loss、Smooth L1、Dice Loss。
- 优化器：SGD、Momentum、Adam、AdamW。
- 学习率策略：StepLR、Cosine Annealing、Warmup、OneCycle。
- 正则化：L2/weight decay、dropout、数据增强、early stopping。
- 指标体系：accuracy、precision、recall、F1、ROC/AUC、mAP、IoU、latency、throughput。
- 数据预处理：归一化、标准化、类别编码、缺失值处理、异常值处理。

### 机器人/智驾相关重点

- 传感器数据的训练集构建：图像、视频帧、点云、IMU、里程计、控制指令。
- 时间序列切片：单帧、双帧、多帧窗口、滑动窗口。
- 训练数据和部署数据分布不一致的问题。
- 模型评估不能只看 loss，还要看实时性、稳定性、边界场景表现。
- 数据标注质量、类别不平衡、长尾场景。

### 实践任务

- 用 sklearn 或 PyTorch 实现一个二分类/回归小任务，完整走通数据划分、训练、验证、指标输出。
- 对一个真实或公开小数据集做 baseline、调参、记录实验表。
- 写一份实验记录：数据来源、模型、loss、指标、错误样本分析。

## 2.2 深度学习

### 基础模块

- MLP：线性层、激活函数、隐藏层、表达能力。
- CNN：卷积、padding、stride、dilation、pooling、BN、残差连接。
- 常见视觉 backbone：LeNet、AlexNet、VGG、ResNet、MobileNet、EfficientNet。
- Transformer：token、position embedding、self-attention、cross-attention、FFN、LayerNorm。
- ViT：patch embedding、CLS token、图像分类流程。
- 检测与分割基础：anchor、one-stage/two-stage、YOLO、Mask R-CNN、U-Net、DeepLab。
- 多模态基础：图像 encoder、文本 encoder、特征对齐、contrastive learning、CLIP 思路。

### 训练细节

- batch size、gradient accumulation、梯度裁剪。
- 参数初始化、模型保存、checkpoint resume。
- train/eval 模式区别：Dropout、BatchNorm。
- 混合精度训练：autocast、GradScaler、FP16/BF16 的数值风险。
- 分布式训练概念：DataParallel、DDP、ZeRO、tensor parallel、pipeline parallel，只需理解概念和适用场景。
- profiling：训练耗时拆分为 data loading、forward、backward、optimizer step。

### 机器人/智驾相关重点

- 感知模型：车道线、目标检测、语义分割、深度估计、光流、BEV 感知的基本任务形态。
- 端侧模型选择：MobileNet、ShuffleNet、轻量 CNN、Tiny YOLO。
- 多帧输入：双帧/多帧堆叠、光流、temporal module。
- 模型部署约束：输入尺寸、算子支持、量化误差、延迟预算。

### 实践任务

- 用 PyTorch 训练一个 CNN 分类模型，记录 train/val loss 曲线。
- 改造一个 MobileNet/ResNet 模型，完成迁移学习。
- 导出 ONNX，并用 ONNX Runtime 对比 PyTorch 输出误差。
- 写一个错误样本分析脚本，把预测错的图片保存出来。

## 2.3 PyTorch

### 张量与基础操作

- Tensor 创建、shape、dtype、device。
- view、reshape、permute、transpose、contiguous 的区别。
- broadcasting、indexing、mask、gather、scatter。
- CPU/GPU 数据拷贝：to(device)、pin_memory、non_blocking。
- NumPy 与 Tensor 互转，何时会共享内存。

### Autograd

- requires_grad、grad_fn、leaf tensor。
- backward 的计算图释放机制。
- detach、no_grad、inference_mode。
- 自定义 loss 与梯度检查。
- 梯度爆炸、梯度消失、梯度裁剪。

### Dataset/DataLoader

- Dataset、DataLoader、Sampler。
- collate_fn、自定义 batch。
- num_workers、prefetch_factor、pin_memory。
- 数据增强：torchvision transforms、Albumentations。
- 数据读取瓶颈定位：CPU decode、磁盘 IO、batch 拼接。

### nn.Module 体系

- Module、Parameter、buffer。
- state_dict 保存与加载。
- train/eval 模式。
- hook：forward hook、backward hook。
- 常用层：Conv2d、BatchNorm、LayerNorm、Dropout、Embedding、MultiheadAttention。

### 训练工程

- 标准训练循环：forward、loss、backward、step、zero_grad。
- AMP 混合精度。
- checkpoint 保存与恢复。
- TensorBoard/Weights & Biases 记录。
- profiler 使用：CPU/GPU 时间、算子耗时、显存占用。
- reproducibility：seed、deterministic、cudnn benchmark。

### Transformer 基本实现

- 手写 scaled dot-product attention。
- 实现 MHA，理解 Q/K/V shape。
- 实现一个最小 Transformer Encoder。
- 用小数据集训练一个 toy language model 或分类模型。
- 对照 PyTorch `nn.MultiheadAttention` 和自己实现的版本。

### 实践任务

- 从零写一个 `train.py`：支持配置、日志、checkpoint、resume、eval。
- 手写一个 tiny Transformer，并跑通训练。
- 用 profiler 找出训练瓶颈，写出优化前后对比。

## 2.4 CUDA

### CUDA 编程模型

- host/device 的职责划分。
- kernel、grid、block、thread。
- threadIdx、blockIdx、blockDim、gridDim。
- SIMT 执行模型。
- warp、warp divergence。

### 内存体系

- global memory、shared memory、local memory、constant memory、texture memory。
- 显存访问合并 coalescing。
- shared memory bank conflict。
- register pressure 与 occupancy。
- unified memory 的优缺点。
- pinned memory 与异步拷贝。

### CUDA Runtime API

- cudaMalloc、cudaFree、cudaMemcpy。
- kernel launch。
- cudaGetLastError、cudaDeviceSynchronize。
- cudaEvent 计时。
- stream、event、异步拷贝。
- 多 stream 并发、拷贝和计算重叠。

### 性能优化

- 矩阵加法、向量加法、矩阵乘法的 naive kernel。
- shared memory tiled matmul。
- reduce 操作。
- memory bandwidth vs compute bound。
- Nsight Systems 看时间线。
- Nsight Compute 看 kernel 指标。
- 与 cuBLAS/cuDNN/TensorRT 的关系：实际工程中优先使用成熟库，手写 kernel 用于理解瓶颈和特殊算子。

### 机器人/智驾相关重点

- 图像预处理加速：resize、normalize、layout transform、BGR/RGB 转换。
- batch 和 stream 如何影响端到端延迟。
- CPU/GPU 同步点对实时系统的影响。
- 推理 pipeline 的计时分桶：decode、preprocess、H2D、infer、D2H、postprocess。

### 实践任务

- 写 vector add、matrix transpose、tiled matmul 三个 kernel。
- 用 cudaEvent 测时间，并和 CPU/Numpy/PyTorch 对比。
- 写一个简单图像 normalize kernel。
- 用 Nsight Systems 观察一次 PyTorch 推理或 TensorRT 推理时间线。

## 2.5 TensorRT / ONNX / 模型量化

### ONNX

- ONNX 的作用：训练框架与推理框架之间的中间表示。
- opset、operator、initializer、dynamic axis。
- PyTorch 导出 ONNX：输入 shape、动态 batch、eval 模式。
- ONNX Runtime 推理。
- Netron 查看模型图。
- 常见导出问题：不支持算子、动态控制流、shape 不确定、精度不一致。

### TensorRT 基础

- TensorRT engine、builder、network、parser、execution context。
- FP32、FP16、INT8 推理。
- dynamic shape 与 optimization profile。
- workspace、tactic selection。
- trtexec 的基本使用。
- engine 序列化与加载。
- TensorRT Python API 与 C++ API 的区别。

### 量化

- PTQ 与 QAT。
- 对称/非对称量化。
- per-tensor 与 per-channel。
- scale、zero point。
- calibration dataset 的选择。
- INT8 精度下降定位：逐层对比、敏感层回退、混合精度。
- 和 RKNN/端侧 NPU 量化的共同点：校准数据、算子支持、输入预处理一致性。

### Profiling 与 Debug

- trtexec 查看 latency、throughput、layer time。
- Nsight Systems 看 H2D/D2H 和 kernel timeline。
- Polygraphy 做 ONNX/TensorRT 输出对比。
- 排查模型部署问题：输入布局错误、归一化不一致、动态 shape 配置错误、unsupported op。

### 实践任务

- 训练一个 PyTorch CNN -> 导出 ONNX -> ONNX Runtime 校验 -> TensorRT FP16 engine -> 比较输出误差和速度。
- 准备 calibration 数据，做一次 INT8 PTQ。
- 写一份部署记录：输入 shape、预处理、opset、精度误差、延迟。

## 2.6 ROS/ROS2

### ROS2 核心概念

- workspace、package、node。
- topic、publisher、subscriber。
- service、client。
- action。
- message、srv、action 自定义接口。
- parameter。
- launch file。
- namespace、remapping。
- lifecycle node。
- QoS：reliability、durability、history、depth。

### 工具链

- colcon build。
- ros2 pkg、ros2 node、ros2 topic、ros2 service、ros2 action、ros2 param。
- rqt_graph、rviz2。
- rosbag2 record/play。
- tf2 坐标变换。
- URDF/Xacro 基础。

### C++/Python 节点

- rclcpp/rclpy 基本结构。
- timer callback。
- subscriber callback。
- callback group。
- executor：single-threaded vs multi-threaded。
- 参数读取与动态更新。
- launch 启动多个节点。

### 机器人/智驾相关重点

- Camera/LiDAR/IMU/GPS/Odom 数据流。
- sensor_msgs/Image、PointCloud2、Imu、NavSatFix。
- cv_bridge 图像转换。
- image_transport。
- tf tree：base_link、camera_link、map、odom。
- 数据同步：message_filters、ApproximateTime。
- 实时性问题：队列长度、QoS、callback 阻塞、节点间拷贝。

### 实践任务

- 创建一个 ROS2 workspace 和两个 package。
- 写一个 camera publisher 节点，发布图片。
- 写一个 inference subscriber 节点，订阅图片并跑一个 PyTorch/ONNX Runtime 模型。
- 写 launch 文件同时启动 publisher、inference、rviz2。
- 用 rosbag2 录制和回放图片 topic。
- 写一份节点图和 topic 表。

## 2.7 强化学习

### 基础概念

- agent、environment、state、observation、action、reward。
- policy、value function、Q function。
- return、discount factor。
- exploration vs exploitation。
- on-policy vs off-policy。
- model-free vs model-based。
- imitation learning 与 reinforcement learning 的区别。

### 经典算法

- Dynamic Programming：policy iteration、value iteration。
- Q-learning、SARSA。
- DQN：experience replay、target network。
- Policy Gradient。
- Actor-Critic。
- PPO、SAC。
- Offline RL 的基本问题：distribution shift、conservative learning。

### 机器人相关重点

- continuous action space。
- sim2real gap。
- reward shaping。
- imitation learning：behavior cloning。
- Diffusion Policy、ACT、RT-1/RT-2/VLA 与强化学习的关系。
- 为什么很多机器人策略先从 imitation learning 入门，而不是纯 RL。

### 实践任务

- 用 Gymnasium 跑 CartPole 的 Q-learning/DQN。
- 跑一个 PPO baseline，理解日志里的 reward、episode length、loss。
- 用一个小型行为克隆任务理解 observation/action dataset。
- 写一页总结：RL、IL、VLA 在机器人策略学习里的位置。

## 2.8 VLA

### 基础概念

- VLA = Vision-Language-Action：视觉输入、语言指令、动作输出。
- 与 VLM 的区别：VLM 输出文本，VLA 输出机器人动作或动作 token。
- action representation：离散动作 token、连续动作向量、末端执行器位姿、关节角、gripper。
- policy head：如何从多模态表征映射到动作。
- action chunking：一次预测多步动作。
- open-loop vs closed-loop 控制。
- language grounding：语言指令如何约束动作。

### 代表方向

- RT-1/RT-2：机器人数据、多任务指令、动作 token。
- OpenVLA：开源 VLA，理解数据格式、模型输入输出、finetune 思路。
- Diffusion Policy：用扩散模型生成动作序列。
- ACT：action chunking transformer。
- LeRobot：机器人学习数据集、训练、部署工具链。
- LIBERO/MetaWorld/Robomimic：仿真和评估环境。

### 需要重点理解的问题

- VLA 不是只懂大模型，还需要懂机器人数据闭环。
- 数据格式比模型名字更关键：图像、语言、状态、动作、时间戳。
- 评估指标不是 BLEU/accuracy，而是 task success rate、轨迹稳定性、碰撞、恢复能力。
- 推理延迟会直接影响控制频率。
- 真机部署要考虑安全边界、动作裁剪、急停、坐标系、控制接口。

### 实践任务

- 阅读一个 VLA 项目的 README 和数据格式说明，画出输入输出流程。
- 用 LeRobot 或公开 demo 跑一个 imitation learning 小任务。
- 不追求真机，优先在仿真环境里理解 observation/action。
- 写一篇短笔记：从触觉/视觉感知和端侧部署切入 VLA 的可行路径。

## 2.9 vLLM

### 基础概念

- LLM serving 的核心指标：TTFT、TPOT、latency、throughput、QPS、并发数。
- prefill 与 decode 阶段。
- KV Cache 的显存占用。
- continuous batching。
- PagedAttention。
- tensor parallel、pipeline parallel。
- quantization：AWQ、GPTQ、FP8、INT8/INT4 的基本理解。
- OpenAI-compatible API。

### 工程内容

- 安装和启动 vLLM server。
- 使用 OpenAI Python client 调用本地服务。
- 参数理解：model、dtype、tensor_parallel_size、gpu_memory_utilization、max_model_len、max_num_seqs。
- benchmark：固定 prompt 长度、输出长度、并发数，测吞吐和延迟。
- 观察显存：nvidia-smi、日志、OOM。
- 理解 scheduler 如何影响吞吐/延迟。

### 和机器人/VLA 的关系

- VLA 或多模态机器人系统可能需要策略模型服务化。
- 机器人系统中常见需求是低延迟、稳定、可控，而不只是最大吞吐。
- vLLM 的经验可迁移到大模型服务、边缘服务器推理、实验平台搭建。

### 实践任务

- 本地或云端启动一个小模型 vLLM 服务。
- 写一个压测脚本，比较单请求、并发请求的 TTFT/总延迟。
- 记录不同 max_model_len、并发数对显存和延迟的影响。

## 2.10 SGLang

### 基础概念

- SGLang 的定位：面向 LLM/VLM 推理和程序化调用的服务框架。
- Runtime、server、frontend language。
- RadixAttention/前缀缓存思想。
- continuous batching。
- structured output。
- speculative decoding。
- attention backend。
- OpenAI-compatible API。

### 工程内容

- 启动 SGLang server。
- 用 Python client 或 HTTP 调用。
- 结构化输出约束：JSON/schema。
- 多轮对话、工具调用、批量请求。
- benchmark：吞吐、延迟、显存。
- 和 vLLM 对比：接口、启动参数、日志、性能表现、适合场景。

### 和机器人/VLA 的关系

- 可用于机器人高层任务规划、语言指令解析、结构化任务输出。
- 可作为 VLA 系统外层的语言规划/任务分解服务。
- 对面试 AI infra/LLM serving 方向有直接价值。

### 实践任务

- 复现一个 SGLang 服务启动和调用。
- 写一个结构化输出 demo：输入自然语言任务，输出 JSON 格式子任务。
- 和 vLLM 做同模型、同 prompt、同并发的简单对比。

## 2.11 计算机视觉与智驾感知补充

### 视觉基础

- 相机模型、内参、外参、畸变。
- 坐标系：像素坐标、相机坐标、世界坐标、车体坐标。
- 图像增强、滤波、边缘、光流。
- 双目/深度估计基础。
- 目标检测、语义分割、实例分割。

### 智驾感知

- 车道线检测。
- 目标检测：车辆、行人、骑行者、交通标志。
- BEV 感知基本概念。
- 多传感器融合：camera、LiDAR、radar、IMU。
- tracking：Kalman Filter、SORT/DeepSORT 基本思路。

### 实践任务

- 用 YOLO 或轻量检测模型跑图片/视频推理。
- 记录 preprocessing、inference、postprocessing 耗时。
- 将检测结果发布为 ROS2 topic。

## 2.12 继续补强的工程底座

### C++

- RAII、智能指针、移动语义。
- STL 容器复杂度。
- 多线程：thread、mutex、condition_variable、atomic。
- CMake target、include、link、install。
- C++ 调用 ONNX Runtime/TensorRT/ROS2。

### Linux/操作系统

- 进程、线程、调度、上下文切换。
- 内存：虚拟内存、mmap、page cache。
- 文件 IO、epoll。
- shared memory、message queue。
- 性能工具：top、htop、perf、strace、lsof。

### 网络

- TCP/UDP、HTTP/WebSocket。
- RTSP/RTP 基础。
- 序列化：JSON、Protobuf。
- 机器人系统里的网络延迟、丢包、队列堆积。

### Git

- branch、commit、rebase、merge。
- stash、cherry-pick。
- 写清晰 commit message。
- 用 README 记录实验复现步骤。

## 3. 两个月周计划

## Week 1：PyTorch 训练工程打底

### 目标

建立一个能复用的 PyTorch 训练模板，避免只会 notebook 调 API。

### 学习内容

- Tensor shape、dtype、device、broadcasting。
- Dataset/DataLoader、自定义 collate_fn。
- nn.Module、state_dict、train/eval。
- loss、optimizer、scheduler。
- AMP、checkpoint、TensorBoard。
- profiler 初步使用。

### 实践产出

- `train.py`：训练一个 CNN 分类或回归模型。
- `eval.py`：加载 checkpoint 输出指标。
- `export_onnx.py`：导出 ONNX。
- 一份实验记录：loss 曲线、指标、错误样本。

### 每日安排

- Day 1：Tensor 与 Autograd。
- Day 2：Dataset/DataLoader。
- Day 3：nn.Module 与训练循环。
- Day 4：CNN/MobileNet 迁移学习。
- Day 5：checkpoint、resume、日志。
- Day 6：AMP 与 profiler。
- Day 7：整理模板和 README。

## Week 2：深度学习视觉基础 + ONNX

### 目标

理解机器人/智驾感知模型的基本结构，并完成 PyTorch 到 ONNX 的第一条部署链路。

### 学习内容

- CNN、ResNet、MobileNet。
- detection/segmentation 任务基本区别。
- 图像预处理一致性：resize、crop、normalize、layout。
- ONNX opset、dynamic axis。
- ONNX Runtime 推理与输出对齐。
- Netron 看模型结构。

### 实践产出

- 一个视觉模型训练/finetune demo。
- PyTorch 输出与 ONNX Runtime 输出误差对比。
- 一份模型部署说明：输入输出、预处理、opset、动态维度。

### 每日安排

- Day 1：CNN/ResNet/MobileNet。
- Day 2：检测/分割任务概念。
- Day 3：训练一个视觉 baseline。
- Day 4：导出 ONNX。
- Day 5：ONNX Runtime 校验。
- Day 6：Netron + 模型图分析。
- Day 7：整理部署链路。

## Week 3：TensorRT 与模型量化

### 目标

把模型部署能力从“能导出”推进到“能加速、能测、能解释误差”。

### 学习内容

- TensorRT engine、builder、context。
- trtexec。
- FP16 engine。
- dynamic shape 与 optimization profile。
- INT8 PTQ、calibration dataset。
- layer-wise profiling。
- 常见部署错误排查。

### 实践产出

- ONNX -> TensorRT FP16 engine。
- 可重复运行的推理脚本。
- latency/throughput 对比表：PyTorch、ONNX Runtime、TensorRT。
- INT8 尝试记录：精度变化和可能原因。

### 每日安排

- Day 1：TensorRT 基本概念和 trtexec。
- Day 2：构建 FP16 engine。
- Day 3：Python API 推理。
- Day 4：dynamic shape。
- Day 5：INT8 PTQ。
- Day 6：profiling 和误差定位。
- Day 7：整理部署报告。

## Week 4：CUDA 编程与 GPU 性能理解

### 目标

不追求写复杂算子，但要真正理解 GPU 为什么快、什么时候不快、如何定位瓶颈。

### 学习内容

- grid/block/thread。
- warp 和 divergence。
- global/shared memory。
- coalescing、bank conflict。
- stream/event。
- cudaEvent 计时。
- Nsight Systems 时间线。

### 实践产出

- vector add kernel。
- tiled matmul kernel。
- image normalize kernel。
- 一份 CUDA 性能笔记：瓶颈、优化点、与 PyTorch/TensorRT 的关系。

### 每日安排

- Day 1：CUDA 编程模型。
- Day 2：内存层级。
- Day 3：vector add + 计时。
- Day 4：matrix transpose/matmul。
- Day 5：shared memory 优化。
- Day 6：stream/event + Nsight。
- Day 7：整理 GPU 性能笔记。

## Week 5：ROS2 系统集成

### 目标

把模型推理接入机器人系统，让你从“模型训练/部署”走到“机器人软件链路”。

### 学习内容

- workspace、package、node。
- topic/service/action。
- message 和自定义接口。
- parameter、launch。
- QoS。
- rosbag2。
- rviz2、rqt_graph。
- tf2。

### 实践产出

- 一个 ROS2 图像发布节点。
- 一个 ROS2 推理订阅节点。
- 一个 launch 文件。
- rosbag2 录制和回放。
- 节点图、topic 表、延迟统计。

### 每日安排

- Day 1：ROS2 基础命令和 workspace。
- Day 2：publisher/subscriber。
- Day 3：service/action/parameter。
- Day 4：launch 和 QoS。
- Day 5：图像 topic + cv_bridge。
- Day 6：模型推理节点。
- Day 7：rosbag2 + README。

## Week 6：机器人/智驾感知小项目

### 目标

做一个能放到简历/面试叙事里的小闭环：视频/图像输入 -> 模型推理 -> ROS2 发布 -> 延迟分析。

### 项目建议

选择一个即可：

- 轻量目标检测：摄像头/视频输入，输出检测框。
- 车道线/道路区域分割：输出 mask。
- 图像分类/状态识别：输出类别和置信度。
- 双帧/多帧输入回归：模拟机器人传感器状态估计。

### 学习内容

- 端到端 pipeline 设计。
- 预处理、推理、后处理计时。
- ROS2 topic 输出结果。
- 可视化结果。
- 延迟优化：batch、输入尺寸、FP16、减少拷贝。

### 实践产出

- 一个完整 demo。
- 一张架构图。
- 一张性能表。
- 一份 README：环境、运行命令、结果截图、后续优化。

### 每日安排

- Day 1：选题和数据准备。
- Day 2：模型 baseline。
- Day 3：ONNX/TensorRT 部署。
- Day 4：ROS2 节点接入。
- Day 5：可视化与 rosbag2。
- Day 6：性能分析和优化。
- Day 7：整理项目文档。

## Week 7：强化学习与 VLA 入门

### 目标

理解机器人策略学习的地图：RL、IL、Diffusion Policy、ACT、VLA 分别在什么位置。

### 学习内容

- MDP、policy、value、Q function。
- DQN、PPO、SAC 的基本思想。
- imitation learning、behavior cloning。
- continuous control。
- VLA 输入输出格式。
- action token/action vector。
- OpenVLA/LeRobot/LIBERO 的项目结构和数据格式。

### 实践产出

- 跑通一个 Gymnasium RL baseline。
- 跑通或阅读一个 imitation learning/VLA demo。
- 画出 VLA 推理流程图。
- 写一份 VLA 切入路线笔记。

### 每日安排

- Day 1：RL 基础概念。
- Day 2：DQN/PPO。
- Day 3：跑一个 RL baseline。
- Day 4：imitation learning。
- Day 5：VLA 概念和代表项目。
- Day 6：阅读 OpenVLA/LeRobot 数据格式。
- Day 7：整理 VLA 笔记。

## Week 8：vLLM/SGLang + 总项目整理

### 目标

把 LLM serving 的基础补上，并把前 7 周内容整合成可展示的学习成果。

### 学习内容

- prefill/decode。
- KV Cache。
- continuous batching。
- PagedAttention。
- vLLM OpenAI-compatible server。
- SGLang server、structured output。
- benchmark：并发、吞吐、延迟、显存。
- LLM 服务与机器人高层规划/VLA 的关系。

### 实践产出

- vLLM 本地/云端服务 demo。
- SGLang 结构化输出 demo。
- vLLM vs SGLang 简单对比表。
- 两个月学习总结和项目索引。

### 每日安排

- Day 1：LLM serving 指标。
- Day 2：vLLM 启动和调用。
- Day 3：vLLM benchmark。
- Day 4：SGLang 启动和调用。
- Day 5：structured output demo。
- Day 6：对比实验。
- Day 7：整理总 README 和简历话术。

## 4. 每周时间分配建议

如果每天能投入 3-4 小时：

| 模块 | 每周时间 | 说明 |
| --- | --- | --- |
| 理论学习 | 6-8 小时 | 看官方文档、课程、论文/博客 |
| 代码实践 | 10-14 小时 | 必须有可运行代码 |
| 实验记录 | 2-3 小时 | 写 README、表格、踩坑记录 |
| 复盘补漏 | 2-3 小时 | 整理问题、二刷关键概念 |

每天建议结构：

- 30 分钟：复习昨天内容。
- 60-90 分钟：学习新概念。
- 90-120 分钟：写代码/跑实验。
- 20 分钟：记录今天产出和问题。

## 5. 两个月最终交付物

建议最后形成一个 `robotics-ai-learning` 仓库，包含：

```text
robotics-ai-learning/
  pytorch_train_template/
    train.py
    eval.py
    export_onnx.py
    README.md
  onnx_tensorrt_demo/
    build_engine.py
    infer_trt.py
    benchmark.md
  cuda_kernels/
    vector_add.cu
    tiled_matmul.cu
    image_normalize.cu
    notes.md
  ros2_inference_demo/
    src/
    launch/
    README.md
  rl_vla_notes/
    rl_baseline.md
    vla_data_flow.md
  llm_serving/
    vllm_benchmark.py
    sglang_structured_output.py
    comparison.md
  summary.md
```

## 6. 面试叙事模板

两个月后，你应该能这样讲自己的方向：

> 我原来主要做嵌入式 Linux 应用和端侧数据流处理，熟悉 C/C++、Linux、多线程、网络、音视频链路。第二阶段我把能力往机器人和智驾方向扩展，补了 PyTorch 训练、ONNX/TensorRT 部署、CUDA 性能基础和 ROS2 系统集成。我做过一个图像/视频输入到模型推理再到 ROS2 topic 发布的端到端 demo，并且能拆分 preprocessing、H2D、inference、postprocessing 的耗时。对于 VLA 和具身智能，我目前重点从多模态感知、机器人数据格式、动作输出和部署延迟角度切入，而不是只停留在大模型概念。

## 7. 资源入口

- PyTorch 官方教程：https://docs.pytorch.org/tutorials/
- PyTorch Autograd 教程：https://docs.pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html
- CUDA C++ Programming Guide：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- TensorRT 官方文档：https://docs.nvidia.com/deeplearning/tensorrt/latest/
- TensorRT Best Practices：https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html
- ROS2 官方文档：https://docs.ros.org/
- ROS2 Tutorials：https://docs.ros.org/en/rolling/Tutorials.html
- vLLM 文档：https://docs.vllm.ai/
- vLLM OpenAI-compatible server：https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html
- SGLang 文档：https://docs.sglang.ai/
- SGLang Attention Backend：https://docs.sglang.ai/advanced_features/attention_backend.html
- OpenVLA 论文：https://arxiv.org/abs/2406.09246

## 8. 执行原则

- 每周必须有可运行代码，不只看视频或文档。
- 每个模型实验都要记录输入 shape、预处理、指标和耗时。
- 每个部署实验都要记录 PyTorch/ONNX/TensorRT 输出误差。
- 每次性能优化都要先测 baseline，再改动，再复测。
- VLA 不要一开始追求训练大模型，先理解数据格式、策略输出、评估方式和部署链路。
- ROS2 不要只跑 turtlesim，尽快接入图像 topic 和模型推理节点。
- vLLM/SGLang 不要只启动服务，要做并发压测和指标记录。
