# Reinforcement Learning (强化学习)

> 强化学习理论与算法学习笔记

## 内容概览

- 基本概念（Agent、Environment、State、Action、Reward、Policy）
- 马尔可夫决策过程（MDP）
- 值函数方法（Q-Learning、DQN）
- 策略梯度方法（REINFORCE、PPO、SAC、TRPO）
- Actor-Critic 框架
- 模仿学习（Imitation Learning）与离线强化学习（Offline RL）

## 1. 核心概念

强化学习：Agent 通过与 Environment 交互，最大化累积奖励。

```
Agent ──action──► Environment
  ▲                    │
  └──state, reward─────┘
```

| 概念 | 说明 |
|------|------|
| **State (s)** | 当前环境状态 |
| **Action (a)** | Agent 可执行的操作 |
| **Reward (r)** | 执行动作后环境给的即时反馈 |
| **Policy (π)** | 状态到动作的映射：$\pi(a\|s)$ |
| **Value function V(s)** | 从状态 s 出发的期望累积奖励 |
| **Q function Q(s,a)** | 在 s 下执行 a 后的期望累积奖励 |
| **Episode** | 一次完整的交互序列（start → terminal） |
| **Return G** | 折扣累积奖励：$G_t = \sum_{k=0}^{\infty}\gamma^k r_{t+k}$ |

折扣因子 $\gamma \in [0,1]$：控制对未来奖励的重视程度。

---

## 2. 马尔可夫决策过程（MDP）

**马尔可夫性**：未来状态只依赖当前状态（无记忆）。

MDP = $(S, A, P, R, \gamma)$：
- $P(s'|s,a)$：状态转移概率
- $R(s,a)$：奖励函数

**贝尔曼方程**：

$$V^\pi(s) = \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]$$

**最优贝尔曼方程**：

$$Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q^*(s',a')$$

---

## 3. 值函数方法

### Q-Learning（Off-Policy, Model-Free）
TD 更新，直接估计最优 Q 值：
$$Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma\max_{a'}Q(s',a') - Q(s,a)\right]$$

- **ε-greedy 策略**：以概率 ε 随机探索，1-ε 利用已知最优动作
- 表格型：状态空间离散且小时使用 Q-table
- **探索-利用权衡（Exploration-Exploitation Tradeoff）**：核心难题

### DQN（Deep Q-Network）
Q-table → 神经网络，处理高维状态（图像）：
- **经验回放（Experience Replay）**：存储 $(s,a,r,s')$ 到 Replay Buffer，随机采样，打破时序相关性
- **目标网络（Target Network）**：每 N 步同步一次，稳定训练目标

---

## 4. 策略梯度方法

直接优化策略 $\pi_\theta$，梯度：

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi\left[\nabla_\theta \log\pi_\theta(a|s) \cdot Q^\pi(s,a)\right]$$

### REINFORCE
蒙特卡洛策略梯度，用 Return $G_t$ 作为 Q 的估计。

### PPO（Proximal Policy Optimization）
**工业界最常用的策略梯度算法**（OpenAI、DeepMind 广泛使用）：

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

- $r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$ 是重要性比率
- Clip 防止策略更新过大（保守更新）
- 特点：稳定、高效、易调参

### SAC（Soft Actor-Critic）
最大熵强化学习，同时优化奖励和策略熵，适合连续动作空间：
$$J(\pi) = \sum_t \mathbb{E}\left[r_t + \alpha\mathcal{H}(\pi(\cdot|s_t))\right]$$

---

## 5. Actor-Critic 框架

结合值函数（Critic）和策略（Actor）的优势：

```
Actor：  π_θ(a|s)  → 选择动作（策略网络）
Critic： V_φ(s)    → 评估状态价值（值网络）
Advantage: A(s,a) = Q(s,a) - V(s)  → 减少方差
```

**TD3 / SAC / PPO 都是 Actor-Critic 的变体。**

---

## 6. 模仿学习与离线 RL

| 方法 | 说明 | 数据来源 |
|------|------|---------|
| **行为克隆（BC）** | 直接监督学习模仿专家动作 | 专家演示 |
| **GAIL** | 用 GAN 框架模仿专家状态-动作分布 | 专家演示 |
| **Offline RL** | 从静态历史数据中学习，不与环境交互 | 任意历史数据 |
| **IRL（逆强化学习）** | 从专家行为中推断奖励函数 | 专家演示 |

**VLA 模型的训练**通常结合行为克隆（预训练）+ RL 微调（提升泛化）。

```python
import numpy as np
import random
from collections import defaultdict, deque

# ===== 1. Q-Table Q-Learning（FrozenLake 风格的格子世界）=====
# 简单 4x4 格子世界：状态=格子编号，动作=上下左右
# 目标：从起点(0)走到终点(15)，避开陷阱

class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.start = 0
        self.goal = size * size - 1
        self.holes = {5, 7, 11, 12}  # 陷阱格子
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        # 动作: 0=左, 1=下, 2=右, 3=上
        row, col = divmod(self.state, self.size)
        if action == 0 and col > 0: col -= 1
        elif action == 1 and row < self.size-1: row += 1
        elif action == 2 and col < self.size-1: col += 1
        elif action == 3 and row > 0: row -= 1
        self.state = row * self.size + col
        if self.state in self.holes:
            return self.state, -1.0, True
        if self.state == self.goal:
            return self.state, 1.0, True
        return self.state, -0.01, False  # 小惩罚鼓励走最短路

def q_learning(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=1.0, eps_decay=0.995):
    n_states = env.size ** 2
    n_actions = 4
    Q = np.zeros((n_states, n_actions))
    rewards_history = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(100):  # 最多100步
            # ε-greedy 探索
            if random.random() < epsilon:
                action = random.randint(0, n_actions - 1)
            else:
                action = np.argmax(Q[state])

            next_state, reward, done = env.step(action)
            total_reward += reward

            # Q-Learning 更新
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )
            state = next_state
            if done:
                break

        epsilon = max(0.01, epsilon * eps_decay)
        rewards_history.append(total_reward)

    return Q, rewards_history

env = GridWorld()
Q, history = q_learning(env, episodes=3000)
print("Q-Learning 训练完成")
print(f"最后100个episode平均奖励: {np.mean(history[-100:]):.3f}")

# 展示学到的最优策略
action_names = ['←', '↓', '→', '↑']
print("\n最优策略（格子世界4×4）:")
for row in range(4):
    row_str = ""
    for col in range(4):
        s = row * 4 + col
        if s in env.holes: row_str += "  X  "
        elif s == env.goal: row_str += "  G  "
        else: row_str += f"  {action_names[np.argmax(Q[s])]}  "
    print(row_str)

# ===== 2. DQN 结构（PyTorch 实现）=====
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    """Deep Q-Network：将状态映射到每个动作的 Q 值"""
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(states),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(next_states),
                torch.FloatTensor(dones))

    def __len__(self): return len(self.buf)

def dqn_update(online_net, target_net, buffer, optimizer, batch_size=64, gamma=0.99):
    if len(buffer) < batch_size:
        return 0.0
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    # 当前 Q 值
    q_values = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    # 目标 Q 值（用 target_net 计算，更稳定）
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0]
        targets = rewards + gamma * next_q * (1 - dones)

    loss = nn.functional.smooth_l1_loss(q_values, targets)  # Huber Loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
    optimizer.step()
    return loss.item()

# 示例：CartPole 环境（如已安装 gymnasium）
STATE_DIM, ACTION_DIM = 4, 2  # CartPole 状态维度=4，动作=2
online_net = DQN(STATE_DIM, ACTION_DIM)
target_net = DQN(STATE_DIM, ACTION_DIM)
target_net.load_state_dict(online_net.state_dict())
optimizer = optim.Adam(online_net.parameters(), lr=1e-3)
buffer = ReplayBuffer()

# 模拟几步交互
for _ in range(200):
    s = np.random.randn(STATE_DIM).astype(np.float32)
    a = random.randint(0, ACTION_DIM - 1)
    r = random.uniform(-1, 1)
    ns = np.random.randn(STATE_DIM).astype(np.float32)
    d = random.random() < 0.1
    buffer.push(s, a, r, ns, d)

loss = dqn_update(online_net, target_net, buffer, optimizer)
print(f"\nDQN 单步更新 loss: {loss:.4f}")
print("DQN 架构:", online_net)

# ===== 3. PPO 核心 Loss 演示 =====
def ppo_clip_loss(old_log_probs, new_log_probs, advantages, clip_eps=0.2):
    """PPO Clip Loss 计算"""
    ratio = torch.exp(new_log_probs - old_log_probs)          # π_new / π_old
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    return loss

# 模拟数据
old_lp = torch.randn(32)
new_lp = old_lp + torch.randn(32) * 0.1   # 策略更新后的 log probs
adv = torch.randn(32)                        # Advantage 估计

loss_ppo = ppo_clip_loss(old_lp, new_lp, adv)
print(f"\nPPO Clip Loss: {loss_ppo.item():.4f}")

# ===== 4. 实战：gymnasium 环境（如已安装）=====
try:
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    state, _ = env.reset()
    total_r = 0
    for _ in range(200):
        action = env.action_space.sample()   # 随机策略
        state, reward, terminated, truncated, _ = env.step(action)
        total_r += reward
        if terminated or truncated:
            break
    print(f"\nCartPole 随机策略总奖励: {total_r}")
    print("训练真正的 DQN/PPO 推荐使用 stable-baselines3:")
    print("  pip install stable-baselines3")
    print("  from stable_baselines3 import PPO")
    print("  model = PPO('MlpPolicy', env, verbose=1)")
    print("  model.learn(total_timesteps=100_000)")
except ImportError:
    print("\n安装 gymnasium: pip install gymnasium")
    print("完整 DQN/PPO 训练推荐: pip install stable-baselines3")
```
