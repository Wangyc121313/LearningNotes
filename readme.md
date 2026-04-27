# 📚 Learning Notes

> 个人技术学习笔记仓库，涵盖编程语言、工具链、AI/ML 等多个领域。

---

## 📂 目录结构

### 💻 编程语言

| 分类 | 内容 | 格式 |
|------|------|------|
| 🔵 [C](./C/C.ipynb) | C 语言基础：指针、内存、数据结构 | Jupyter Notebook |
| 🔷 [C++](./Cpp/Cpp.ipynb) | C++ 语言进阶 | Jupyter Notebook |
| 🐍 [Python](./Python/) | 语法、NumPy、Matplotlib、Pandas、PyTorch、TensorFlow、FastAPI、Scikit-learn | Jupyter Notebook |

### 🛠️ 系统与工具

| 分类 | 内容 | 格式 |
|------|------|------|
| 🐧 [Linux](./Linux/notes.ipynb) | Linux 系统使用与命令 | Jupyter Notebook |
| 🐚 [Shell](./Shell/notes.ipynb) | Shell 脚本与命令行 | Jupyter Notebook |
| 🏗️ [CMake](./CMake/CMake.ipynb) | 跨平台构建系统 | Jupyter Notebook |
| 🧪 [Conda](./Conda/notes.ipynb) | Python 环境与包管理 | Jupyter Notebook |
| 🐳 [Docker](./Docker/notes.ipynb) | 容器化与镜像管理 | Jupyter Notebook |
| 🌿 [Git](./Git/notes.ipynb) | 版本控制与协作工作流 | Jupyter Notebook |

### 🤖 AI / 机器学习基础

> ML、DL、RL 是 AI 的三大学习范式，各有侧重。

| 分类 | 内容 | 格式 |
|------|------|------|
| 📊 [ML](./ML/notes.ipynb) | 经典机器学习（SVM、决策树、聚类、降维等） | Jupyter Notebook |
| 🧠 [DL](./DL/notes.ipynb) | 深度学习基础（CNN、RNN、Transformer 等） | Jupyter Notebook |
| 🎮 [RL](./RL/notes.ipynb) | 强化学习（Q-Learning、PPO、SAC 等） | Jupyter Notebook |

### 🌐 大模型 / 多模态

> LLM → VLM → VLA，能力逐步扩展：纯语言 → 视觉语言 → 视觉语言动作。

| 分类 | 内容 | 格式 |
|------|------|------|
| 💬 [LLM](./LLM/notes.ipynb) | 大语言模型（GPT、LLaMA、Qwen 等） | Jupyter Notebook |
| 👁️ [VLM](./VLM/notes.ipynb) | 视觉语言模型（CLIP、LLaVA、InternVL、Qwen-VL 等） | Jupyter Notebook |
| 🦾 [VLA](./VLA/notes.ipynb) | 视觉语言动作模型（RT-2、OpenVLA、π0 等） | Jupyter Notebook |

### ⚡ 推理与部署

| 分类 | 内容 | 格式 |
|------|------|------|
| 🔥 [CUDA](./Cuda/notes.ipynb) | NVIDIA CUDA 并行计算 | Jupyter Notebook |
| 📦 [ONNX](./ONNX/notes.ipynb) | 模型交换格式与跨框架推理 | Jupyter Notebook |
| 🚀 [TensorRT](./TensorRT/notes.ipynb) | NVIDIA 高性能推理引擎 | Jupyter Notebook |
| ⚡ [vLLM](./vLLM/notes.ipynb) | 大语言模型高效推理框架（PagedAttention） | Jupyter Notebook |

### 🤖 机器人

| 分类 | 内容 | 格式 |
|------|------|------|
| 🤖 [ROS2](./ROS2/notes.ipynb) | 机器人操作系统 ROS 2 | Jupyter Notebook |

---

## 🗂️ Python 子模块

| 文件 | 内容 |
|------|------|
| [notes.ipynb](./Python/notes.ipynb) | Python 基础语法 |
| [Numpy&Matplotlib&Pandas.ipynb](./Python/Numpy%26Matplotlib%26Pandas.ipynb) | 数据处理与可视化 |
| [Pytorch.ipynb](./Python/Pytorch.ipynb) | 深度学习框架 PyTorch |
| [TensorFlow.ipynb](./Python/TensorFlow.ipynb) | 深度学习框架 TensorFlow |
| [Scikit_learn.ipynb](./Python/Scikit_learn.ipynb) | 机器学习库 Scikit-learn |
| [FastAPI.ipynb](./Python/FastAPI.ipynb) | Web 框架 FastAPI |

---

## 🧩 知识结构说明

### AI 三大学习范式

```
ML（机器学习）—— 从数据中学习规律，经典算法为主
  ├── DL（深度学习）—— 以神经网络为核心，ML 的子集
  └── RL（强化学习）—— 通过与环境交互学习策略，可与 DL 结合（Deep RL）
```

### 大模型演进路线

```
LLM（大语言模型）
  └── 纯文本理解与生成

VLM（视觉语言模型）= LLM + Vision Encoder
  └── 图文理解、视觉问答、图像描述

VLA（视觉语言动作模型）= VLM + Action Head
  └── 机器人端到端感知-规划-执行
```

### 推理部署工具链

```
训练完成的模型
  ├── ONNX 导出（框架无关的中间格式）
  │     └── TensorRT 优化（NVIDIA GPU 高性能推理）
  └── vLLM（专为 LLM/VLM 设计的高吞吐推理服务）
        └── 依赖 CUDA 并行计算底层
```

---

## 🛠️ 环境

- 笔记格式：[Jupyter Notebook](https://jupyter.org/) `.ipynb`
- 推荐使用 [VS Code](https://code.visualstudio.com/) + Jupyter 扩展 或 JupyterLab 打开

---

## 📝 说明

本仓库为个人学习记录，持续更新中。笔记内容来源包括官方文档、菜鸟教程、各类课程等，仅供学习参考。
