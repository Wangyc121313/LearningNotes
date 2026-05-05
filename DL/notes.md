# Deep Learning (深度学习)

> 深度学习理论与实践学习笔记

## 内容概览

- 神经网络基础（前向传播、反向传播、激活函数）
- 卷积神经网络（CNN）
- 循环神经网络（RNN、LSTM、GRU）
- Transformer 架构（Attention、Self-Attention、Multi-Head Attention）
- 正则化与优化方法（Dropout、BatchNorm、Adam）
- 经典模型（ResNet、ViT、BERT）

## 1. 神经网络基础

### 前向传播（Forward Pass）
数据从输入层经隐藏层到输出层，每层计算：

$$\mathbf{z}^{(l)} = W^{(l)} \mathbf{a}^{(l-1)} + b^{(l)}, \quad \mathbf{a}^{(l)} = f(\mathbf{z}^{(l)})$$

### 反向传播（Backpropagation）
利用链式法则计算损失对各参数的梯度：

$$\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial \mathbf{z}^{(l)}} \cdot \frac{\partial \mathbf{z}^{(l)}}{\partial W^{(l)}}$$

PyTorch 中通过 `loss.backward()` 自动完成。

### 常用激活函数

| 函数 | 公式 | 特点 |
|------|------|------|
| Sigmoid | $\sigma(x)=\frac{1}{1+e^{-x}}$ | 输出 [0,1]，梯度消失问题 |
| Tanh | $\tanh(x)$ | 输出 [-1,1]，零中心 |
| **ReLU** | $\max(0,x)$ | 最常用，计算快，可能死亡 |
| Leaky ReLU | $\max(0.01x, x)$ | 解决 ReLU 死亡问题 |
| GELU | $x\cdot\Phi(x)$ | Transformer 常用 |

---

## 2. 卷积神经网络（CNN）

**核心操作**：卷积（局部感受野 + 权重共享）→ 提取空间特征

```
Input → [Conv → BN → ReLU] × N → Pooling → Flatten → FC → Output
```

- **Conv2d**：kernel 在输入上滑动，提取局部特征
- **Pooling**：MaxPool/AvgPool 降采样，提升平移不变性
- **BatchNorm**：归一化激活分布，加速训练

**经典架构**：LeNet → AlexNet → VGG → **ResNet**（残差连接）→ EfficientNet

**残差连接**：$\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$，解决深层网络梯度消失，是现代 CNN 基础。

---

## 3. 循环神经网络（RNN / LSTM / GRU）

处理**序列数据**，隐状态携带上下文信息。

### 原始 RNN
$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$
问题：长序列梯度消失/爆炸。

### LSTM（Long Short-Term Memory）
引入**细胞状态**（cell state）和三个门：

| 门 | 作用 |
|----|------|
| 遗忘门 $f_t$ | 决定遗忘多少旧信息 |
| 输入门 $i_t$ | 决定写入多少新信息 |
| 输出门 $o_t$ | 决定输出多少细胞状态 |

### GRU
LSTM 的简化版（合并遗忘门和输入门），参数更少，效果相近。

---

## 4. Transformer 架构

**完全基于 Attention，抛弃 RNN**，支持并行训练，成为 LLM/VLM/VLA 的基础。

### Self-Attention

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

每个 token 与所有其他 token 计算相关性权重，再加权聚合。

### Multi-Head Attention
并行运行 h 组 Attention，捕捉不同语义维度的关系：

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O$$

### Transformer Block 结构
```
x → LayerNorm → Multi-Head Attention → (+residual) → LayerNorm → FFN → (+residual)
```

---

## 5. 正则化与优化

### 正则化
| 方法 | 说明 |
|------|------|
| **Dropout** | 训练时随机置零神经元，防止过拟合 |
| **BatchNorm** | 归一化每层输出，加速收敛 |
| **LayerNorm** | 对每个样本归一化，NLP/Transformer 常用 |
| L1/L2 正则 | 在损失上加权重惩罚项 |

### 优化器对比

| 优化器 | 特点 | 适用场景 |
|--------|------|---------|
| SGD | 基础，有动量版本 | CV 模型精调 |
| Adam | 自适应学习率，收敛快 | 通用首选 |
| AdamW | Adam + 正确的权重衰减 | Transformer 训练标准 |
| LAMB | 大 batch 分布式训练 | BERT 预训练 |

---

## 6. 经典模型速查

| 模型 | 年份 | 贡献 |
|------|------|------|
| ResNet | 2015 | 残差连接，可训练 1000+ 层 |
| BERT | 2018 | 双向 Transformer 预训练，NLP 基础 |
| GPT | 2018 | 单向 Transformer，自回归生成 |
| ViT | 2020 | 将图像切 Patch 后用 Transformer 处理 |
| CLIP | 2021 | 图文对比学习，连接视觉与语言 |

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ===== 1. 手写 MLP（多层感知机）=====
class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=256, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_dim)
        )
    def forward(self, x):
        return self.net(x)

model = MLP().to(device)
print(f"\nMLP 参数量: {sum(p.numel() for p in model.parameters()):,}")

# ===== 2. 模拟训练循环 =====
X = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

for epoch in range(3):
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()           # 清零梯度
        logits = model(xb)              # 前向传播
        loss = criterion(logits, yb)    # 计算损失
        loss.backward()                 # 反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()                # 更新参数
        total_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

# ===== 3. CNN 示例（图像分类）=====
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28x28 → 28x28
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                               # 28x28 → 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                               # 14x14 → 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

cnn = SimpleCNN().to(device)
dummy = torch.randn(4, 1, 28, 28).to(device)
print(f"\nCNN 输出形状: {cnn(dummy).shape}")  # [4, 10]
print(f"CNN 参数量: {sum(p.numel() for p in cnn.parameters()):,}")

# ===== 4. LSTM 示例（序列分类）=====
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=64, hidden=128, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=2,
                            batch_first=True, dropout=0.3, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, num_classes)  # *2 for bidirectional
    def forward(self, x):
        x = self.embedding(x)                    # [B, T] → [B, T, E]
        out, (h, c) = self.lstm(x)               # [B, T, 2H]
        last = out[:, -1, :]                      # 取最后时刻
        return self.fc(last)

lstm_model = LSTMClassifier().to(device)
seq = torch.randint(0, 1000, (8, 50)).to(device)  # batch=8, seq_len=50
print(f"\nLSTM 输出形状: {lstm_model(seq).shape}")  # [8, 5]

# ===== 5. Transformer Self-Attention 实现 =====
class SelfAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, T, D]
        attn_out, weights = self.attn(x, x, x)
        x = self.norm(x + attn_out)            # 残差 + LayerNorm
        x = self.norm2(x + self.ffn(x))        # FFN + 残差
        return x, weights

sa = SelfAttention().to(device)
tokens = torch.randn(2, 10, 128).to(device)    # batch=2, seq=10, d_model=128
out, weights = sa(tokens)
print(f"\nSelf-Attention 输出: {out.shape}, 注意力权重: {weights.shape}")
```
