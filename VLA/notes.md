# VLA — Vision-Language-Action Model (视觉语言动作模型)

> 具身智能 / 机器人基础模型学习笔记：在 VLM 基础上增加动作决策能力

## 与 LLM / VLM 的关系

```
LLM  ──────────►  VLM  ──────────►  VLA
（文本）         （文本 + 视觉）    （文本 + 视觉 + 动作）
```

## 内容概览

- VLA 架构：感知（Vision）→ 理解（Language）→ 决策（Action）
- 代表模型：RT-2、OpenVLA、π0（pi-zero）、Octo
- 训练方式：模仿学习、强化学习微调
- 与 ROS2 的结合：将 VLA 输出的动作指令部署到真实机器人
- 数据集：Open X-Embodiment、DROID

## 1. VLA 架构详解

VLA（Vision-Language-Action）= 感知 + 理解 + 决策，端到端地将图像和语言指令映射到机器人动作。

```
输入：相机图像（RGB/D）+ 自然语言指令 + 机器人状态（关节角等）
         │
         ▼
 ┌───────────────────────────────────┐
 │  Vision Encoder（视觉编码器）      │  ← 通常是 ViT 或 ResNet
 │  提取视觉特征 token               │
 └───────────────────┬───────────────┘
                     │ 视觉 token
 ┌───────────────────▼───────────────┐
 │  Language Model（大型 Transformer）│  ← VLM 主干（如 LLaMA、PaLI）
 │  融合视觉 + 语言信息              │
 └───────────────────┬───────────────┘
                     │ 融合特征
 ┌───────────────────▼───────────────┐
 │  Action Head（动作解码器）         │  ← 预测连续动作（位置/速度/力）
 │  输出：∆pose / joint angles / ...  │
 └───────────────────────────────────┘
```

### 动作表示方式

| 方式 | 说明 | 典型维度 |
|------|------|---------|
| **End-effector delta** | 末端执行器位姿增量（ΔX,ΔY,ΔZ,ΔRx,ΔRy,ΔRz + gripper） | 7D |
| **Joint angles** | 各关节角度绝对值或增量 | 6-7D |
| **EE absolute pose** | 末端执行器绝对位姿 | 6D+1 |
| **Tokenized action** | 动作离散化后用 token 表示（如 RT-2） | 依赖 bin 数 |

---

## 2. 代表模型对比

| 模型 | 机构 | 基础模型 | 动作表示 | 特点 |
|------|------|---------|---------|------|
| **RT-2** | Google DeepMind | PaLI-X/Gemini | 离散 token | 首个将 VLM 直接用于机器人的工作 |
| **OpenVLA** | Stanford | Llama 2 7B | 离散 token | 完全开源（模型+代码+数据）|
| **π0 (pi-zero)** | Physical Intelligence | PaliGemma + flow matching | 连续动作流 | 高频动作控制，泛化能力强 |
| **Octo** | Berkeley | Transformer | Diffusion head | 开源，支持多机器人形态 |
| **RoboFlamingo** | ByteDance | OpenFlamingo | 连续 | 基于 in-context learning |
| **CogACT** | THU | CogVLM | Flow | 动作与认知分离设计 |

---

## 3. 训练范式

### 阶段一：大规模预训练（Internet Scale）
- 在图文数据、视频上预训练 VLM 主干（学习通用视觉语言理解）
- 数据：ImageNet、CC3M、LAION、视频描述等

### 阶段二：机器人数据微调
- 在机器人演示数据上监督学习（行为克隆，Behavioral Cloning）
- 损失：$L = \|a_{\text{pred}} - a_{\text{gt}}\|^2$（MSE）或 cross-entropy（离散 token）
- 数据：需要大量高质量 (image, instruction, action) triplet

### 阶段三（可选）：在线 RL 微调
- 在真实或仿真环境中收集数据，用 RL 提升泛化
- 挑战：样本效率低、奖励难以定义

---

## 4. 数据集

| 数据集 | 规模 | 特点 |
|--------|------|------|
| **Open X-Embodiment** | ~1M episodes, 22+ 机器人 | 谷歌主导，最大规模跨机器人数据集 |
| **DROID** | ~76K episodes | Stanford，多样场景 |
| **RLBench** | 仿真任务 | 100 种任务，仿真无需真实机器人 |
| **BridgeData V2** | ~60K episodes | Berkeley，WidowX 机器人，开源 |

---

## 5. 与 ROS 2 的集成

```
VLA 推理节点（Python）                  ROS 2 框架
┌─────────────────────────┐            ┌─────────────────────────┐
│ 1. 订阅相机话题          │◄── /camera ─┤  相机驱动节点             │
│    /camera/image_raw     │            └─────────────────────────┘
│                         │
│ 2. 接收任务指令          │◄── /task   ─  用户/规划器
│    "pick up the cup"     │
│                         │
│ 3. VLA 推理             │
│    image + text → action │
│                         │
│ 4. 发布动作指令          │──► /arm_cmd ─┐
│    geometry_msgs/Twist   │              │  ┌────────────────────┐
│    or JointTrajectory    │              └─►│  机械臂控制节点     │
└─────────────────────────┘                 └────────────────────┘
```

关键 ROS 2 消息类型：
- `sensor_msgs/Image` — 相机图像
- `trajectory_msgs/JointTrajectory` — 关节轨迹
- `geometry_msgs/PoseStamped` — 末端执行器目标位姿

```python
"""
VLA 推理示例
使用 OpenVLA（完全开源，基于 Llama 2 7B + ViT）
安装: pip install transformers accelerate pillow
模型: openvla/openvla-7b（HuggingFace Hub）
"""
import numpy as np
from PIL import Image

# ===== 1. OpenVLA 推理（完整示例）=====
try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
    import torch

    # 加载模型（首次运行会下载 ~15GB，需要 GPU）
    print("加载 OpenVLA 模型...")
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True
    )
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        attn_implementation="flash_attention_2",   # 需要 flash-attn
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to("cuda")

    # 模拟输入：相机图像 + 语言指令
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    instruction = "pick up the red cup and place it on the tray"

    # 推理
    inputs = processor(instruction, image).to("cuda", dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    # action: 7D numpy array [ΔX, ΔY, ΔZ, ΔRx, ΔRy, ΔRz, gripper]

    print(f"预测动作 (7D): {action}")
    print(f"末端执行器位移: Δx={action[0]:.3f}, Δy={action[1]:.3f}, Δz={action[2]:.3f}")
    print(f"旋转: Δrx={action[3]:.3f}, Δry={action[4]:.3f}, Δrz={action[5]:.3f}")
    print(f"夹爪: {action[6]:.3f} ({'open' if action[6] > 0.5 else 'close'})")

except ImportError:
    print("需要安装依赖: pip install transformers accelerate")
    print("模型需要 GPU 和至少 16GB 显存（7B 参数 bfloat16）")

# ===== 2. 模拟 VLA 推理流程（无需 GPU）=====
print("\n===== 模拟 VLA 推理流程 =====")

class MockVLAInference:
    """模拟 VLA 推理流程，说明各步骤数据变换"""
    def __init__(self):
        import torch.nn as nn
        import torch
        # 模拟各组件的输出维度
        self.vision_encoder_dim = 768    # ViT patch tokens
        self.lm_hidden_dim = 4096        # LLM hidden size (7B)
        self.action_dim = 7              # 7D delta action

    def encode_image(self, image_np):
        """Vision Encoder: 224x224x3 → num_patches × 768"""
        H, W = 224, 224
        patch_size = 16
        num_patches = (H // patch_size) * (W // patch_size)  # 196 patches
        vision_tokens = np.random.randn(num_patches, self.vision_encoder_dim)
        print(f"  图像: {image_np.shape} → 视觉 tokens: {vision_tokens.shape}")
        return vision_tokens

    def encode_text(self, text):
        """Tokenizer: text → token_ids"""
        # 模拟分词
        tokens = text.split()
        token_ids = np.random.randint(0, 32000, len(tokens))
        print(f"  文本: '{text[:40]}...' → {len(token_ids)} tokens")
        return token_ids

    def lm_forward(self, vision_tokens, text_tokens):
        """LLM: (vision tokens + text tokens) → hidden states"""
        total_tokens = len(vision_tokens) + len(text_tokens)
        hidden = np.random.randn(total_tokens, self.lm_hidden_dim)
        print(f"  LLM 输入: {total_tokens} tokens → 隐藏状态: {hidden.shape}")
        return hidden

    def action_head(self, hidden_states):
        """Action Head: hidden states → 7D action"""
        # 取最后几个 token 的平均（或专用 action token）
        action_feature = hidden_states[-8:].mean(0)   # 取最后8个token均值
        # 线性投影到动作空间
        action = np.tanh(np.random.randn(self.action_dim)) * 0.05  # 小幅增量
        print(f"  动作特征: {action_feature.shape} → 动作: {action.shape}")
        return action

    def predict(self, image_np, instruction):
        print(f"\n[VLA 推理步骤]")
        v_tokens = self.encode_image(image_np)
        t_tokens = self.encode_text(instruction)
        hidden = self.lm_forward(v_tokens, t_tokens)
        action = self.action_head(hidden)
        return action

mock_vla = MockVLAInference()
dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
action = mock_vla.predict(dummy_image, "grasp the blue block and stack it on the red block")

print(f"\n预测动作 (7D delta):")
labels = ['Δx(m)', 'Δy(m)', 'Δz(m)', 'Δrx(rad)', 'Δry(rad)', 'Δrz(rad)', 'gripper']
for label, val in zip(labels, action):
    print(f"  {label:12s}: {val:+.4f}")

# ===== 3. ROS 2 集成代码片段 =====
print("\n===== ROS 2 集成示意 =====")
ros2_code = '''
# ros2_vla_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
import numpy as np

class VLANode(Node):
    def __init__(self):
        super().__init__('vla_inference_node')
        self.bridge = CvBridge()
        self.latest_image = None

        # 订阅相机图像和任务指令
        self.create_subscription(Image, '/camera/color/image_raw',
                                 self.image_callback, 10)
        self.create_subscription(String, '/task_instruction',
                                 self.task_callback, 10)

        # 发布动作指令
        self.action_pub = self.create_publisher(JointTrajectory,
                                                '/arm_controller/joint_trajectory', 10)

        # 加载 VLA 模型（实际使用时加载 OpenVLA/π0 等）
        self.vla = load_vla_model()
        self.get_logger().info("VLA Node 启动完成")

    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

    def task_callback(self, msg):
        if self.latest_image is None:
            self.get_logger().warn("尚未收到图像")
            return

        instruction = msg.data
        self.get_logger().info(f"执行任务: {instruction}")

        # VLA 推理
        action = self.vla.predict(self.latest_image, instruction)
        # action: [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]

        # 转换为 JointTrajectory 发布（实际需要 IK 求解）
        traj = JointTrajectory()
        traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        point = JointTrajectoryPoint()
        point.positions = action[:6].tolist()
        point.time_from_start.sec = 1
        traj.points = [point]
        self.action_pub.publish(traj)

def main():
    rclpy.init()
    node = VLANode()
    rclpy.spin(node)
    rclpy.shutdown()
'''
print(ros2_code)
```
