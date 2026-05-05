# AGENTS.md — LearningNotes 开发指南

本文件用于指导 AI Agent（如 OpenAI Codex）在该仓库中进行开发和内容维护。

---

## 仓库概述

这是一个**个人技术速成笔记仓库**，以 Markdown（`.md`）作为长期维护与归档主格式，以 Jupyter Notebook（`.ipynb`）作为实验、调试和可运行演示载体。内容覆盖编程语言、系统工具、AI/ML、大模型、推理部署、机器人等领域。目标是"速查速学"，而非面面俱到。

---

## 目录结构

```
LearningNotes/
├── readme.md                  ← 主索引，维护所有笔记的导航链接
├── C/notes.md                 ← 主笔记
├── C/C.ipynb                  ← 可选实验/历史 notebook
├── Cpp/notes.md
├── Cpp/Cpp.ipynb
├── CMake/notes.md
├── CMake/CMake.ipynb
├── Conda/notes.md
├── Conda/notes.ipynb
├── ...
├── Python/                    ← 含多个子主题 md + notebook
│   ├── notes.md
│   ├── notes.ipynb
│   ├── Pytorch.md
│   └── Pytorch.ipynb
├── vLLM/notes.md
├── vLLM/notes.ipynb
├── others.md
└── others.ipynb
```

---

## Markdown 主笔记规范

### 适用范围

- **优先维护 `.md`**：概念解释、面试速记、命令速查、对比表格、流程图、坑点总结、部署步骤。
- **保留 `.ipynb`**：交互式调试、tensor shape 验证、绘图、模型推理实验、需要 kernel 执行的学习过程。
- AI Agent 修改笔记时，默认修改 `.md`；只有用户明确要求可运行 notebook 或需要调试代码时，才修改 `.ipynb`。

### 推荐结构

每个主题的 `notes.md` 通常包含：

```markdown
# 主题名称

## 内容概览
- 核心概念 A
- 核心概念 B
- 实践工具 C

## 1. 理论详解

## 2. 常用命令 / 代码模板

## 3. 面试速记 / 常见问题

## 4. 常见坑点
```

格式要求：

- **语言**：中文说明为主，代码标识符使用英文。
- **表格**：对比类内容优先使用 Markdown 表格。
- **代码块语言标注**：必须写明 `bash`、`python`、`yaml`、`dockerfile`、`text` 等。
- **结构图**：优先使用 ASCII 图或 Mermaid；简单流程可用 `text` 代码块。
- **图片**：引用同目录 `resources/` 子目录，格式：
  ```markdown
  <p align="center">
    <img src="resources/1.png" width="60%">
  </p>
  ```
- **内容风格**：面向速查，避免写成长篇教材；优先保留关键结论、必要背景、最小可用示例。

---

## Notebook 内容规范

### 标准 Cell 结构

Notebook 用于实验和调试，不再作为长期归档主格式。每个 `notes.ipynb` 或子主题 notebook 建议由以下几个 Cell 组成：

1. **Cell 1 — 标题与概览**（Markdown）

   ```markdown
   # 主题名称

   ## 内容概览
   - 核心概念 A
   - 核心概念 B
   - 实践工具 C
   ```

2. **Cell 2 — 理论详解**（Markdown）

   按章节展开，包含：
   - 层级标题（`##` / `###`）
   - 概念解释（中文，简洁直接）
   - 对比表格（`| 特性 | A | B |` 格式）
   - ASCII 结构图或流程图（用 ` ``` ` 包裹的纯文本框图）
   - 关键命令或配置代码块（带语言标注，如 ` ```bash ` ` ```python ` ` ```yaml `）

3. **Cell 3 — 代码演示**（Python）

   可运行的 Python 代码，要求：
   - 涵盖理论 Cell 中的核心用法
   - 对依赖库不可用的情况做 `try/except ImportError` 降级处理
   - 添加清晰的 `# ===== 章节名 =====` 分隔注释
   - Windows 环境兼容（避免硬编码 Linux 路径，bash 命令用 `subprocess` 封装并判断平台）

### Notebook 格式约定

- **语言**：全中文注释和说明，代码本身用英文标识符
- **表格**：对比类内容优先用 Markdown 表格呈现
- **代码块语言标注**：必须写明（`bash`、`python`、`yaml`、`dockerfile` 等）
- **图片**：引用 `resources/` 子目录下的图片，格式：
  ```markdown
  <p align="center">
    <img src="resources/1.png" width="60%">
  </p>
  ```
- **Cell 数量**：保持精简，优先保留调试和验证所需内容；归档型说明应沉淀到 `.md`

---

## readme.md 维护规范

- 新增主题时，必须同步更新 `readme.md` 中对应分类的表格
- 表格格式：`| emoji [主题](./目录/notes.md) | 内容简述 | Markdown / Jupyter Notebook |`
- 如果主题有配套 notebook，可在对应目录保留同名 `.ipynb`，但 `readme.md` 主链接优先指向 `.md`
- 分类已固定（见下），不要随意新增顶级分类：
  - 💻 编程语言
  - 🛠️ 系统与工具
  - 🤖 AI / 机器学习基础（ML / DL / RL）
  - 🌐 大模型 / 多模态（LLM → VLM → VLA）
  - ⚡ 推理与部署（CUDA / ONNX / TensorRT / vLLM）
  - 🤖 机器人（ROS2）

---

## 新增主题流程

1. 在根目录创建新文件夹（如 `Triton/`）
2. 在文件夹内创建 `notes.md`，结构遵循"Markdown 主笔记规范"
3. 如需可运行实验，再创建 `notes.ipynb`
4. 如有图片资源，放入 `Triton/resources/`
5. 更新 `readme.md` 对应分类表格，补充条目

---

## 修改已有笔记

- **追加内容**：优先追加到对应主题的 `.md` 章节中；若是实验过程，再追加到 notebook
- **修订内容**：直接编辑 `.md` 对应章节，保持原有知识结构不变
- **Notebook 修订**：保留可运行调试逻辑，避免把大段归档说明继续塞进 notebook
- **禁止**：删除已有 notebook 的有效实验内容、随意更改 `readme.md` 的分类体系、将多个主题混进同一文件

---

## 代码风格要求

```python
# ✅ 推荐写法
import sys

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ℹ️  PyTorch 未安装，跳过相关示例")

# ===== 章节标题 =====
print("=" * 50)
print("1. 核心概念演示")
print("=" * 50)

if TORCH_AVAILABLE:
    # 实际代码
    ...
else:
    # Python 等效逻辑 / 占位提示
    print("需要安装 torch 才能运行此示例")
```

---

## 禁止行为

- 不要创建零散 `.py` 作为笔记；需要代码演示时放入 `.md` 代码块或 notebook
- 不要在 notebook 中插入大段归档型说明或无注释代码块
- 不要修改 `readme.md` 的整体结构（知识结构说明、分类体系）
- 不要在 `others.md` / `others.ipynb` 中写主题性笔记，该文件用于零散知识点
- 不要在代码 Cell 中执行网络请求或下载模型权重（会导致 CI 超时）
