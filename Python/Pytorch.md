## Pytorch

PyTorch 是一个用于利用 GPU 和 CPU 进行深度学习优化的张量库。

```python
import torch
import torch.optim as optim
import torch.nn as nn

print(torch.__version__)
print(torch.cuda.is_available())
x = torch.rand(3,3)
print(x)
```

#### 1.Tensor和矩阵乘法

Tensore是PyTorch中最基础的数据结构，而矩阵乘法是深度学习模型的核心。

```python
# 标量 (0维)
scalar = torch.tensor(3.14)
print(f"标量: {scalar}, shape: {scalar.shape}")
# 向量 (1维)
vector = torch.tensor([1, 2, 3, 4])
print(f"向量: {vector}, shape: {vector.shape}")
# 矩阵 (2维)
matrix = torch.tensor([[1, 2], [3, 4]])
print(f"矩阵: {matrix}, shape: {matrix.shape}")
# 高维Tensor (常用于输入输出/激活)
high_dim = torch.randn(2, 3, 4, 5)
print(f"高维Tensor: shape: {high_dim.shape}")
```

```python
# 矩阵乘法操作
X = torch.randn(2,3)
print(X)
W = torch.randn(3,4)
print(W)
b = torch.randn(4)
print(b)
# 方法1：@
y1 = X @ W + b
# 方法2：torch.matmul
y2 = torch.matmul(X, W) + b
# 方法3：tensor.matmul
y3 = X.matmul(W) + b
print(y3)

torch.allclose(y1, y2) and torch.allclose(y2, y3)
```

#### 2.einsum操作

使用爱因斯坦求和约定，可以直观地描述张量运算。

```python
X = torch.randn(2, 3)
W = torch.randn(3, 4)
# 数学形式: Y_{be} = sum_d X_{bd} * W_{de}
Y = torch.einsum("bd,de->be", X, W)
print(Y.shape)
```

```python
# 批量矩阵乘法
A = torch.randn(10, 3, 4)
B = torch.randn(10, 4, 5)
# 数学形式: Y_{bij} = sum_k A_{bik} * B_{bkj}
C = torch.einsum("bik,bkj->bij", A, B)
# 等价于
D = torch.bmm(A, B) # 需保证两者的shape为[batch, n, m]和[batch, m, p]
print(C.shape, D.shape)
```

#### 3.元素乘法

```python
A = torch.randn(2, 3)
print(A)
B = torch.randn(2, 3)
print(B)
# 方法1：*
C = A * B
print(C)
# 方法2：einsum
D = torch.einsum("ij,ij->ij", A, B)
print(torch.allclose(C, D))

# 点积
a = torch.tensor([1, 2, 3])
b = torch.tensor([2, 3, 5])
dot_product = torch.einsum("i,i->", a, b)
dot_product2 = torch.sum(a*b)
print(dot_product, dot_product2)
```

#### 4.nn.linear层

``nn.linear``是pytorch最基本的模块之一，其功能是实现一次线性变换：$y=x A^{\top} + b$，其中：
 * x:输入张量
 * A:权重矩阵(out_features × in_features)
 * b:偏置向量(out_features)
 * y:输出张量

```python
linear_layer = nn.Linear(in_features=3, out_features=4, bias=True)
print(f"线性层: {linear_layer}")
print(f"权重shape: {linear_layer.weight.shape}")
print(f"偏置shape: {linear_layer.bias.shape}")
print(f"权重: {linear_layer.weight.data}")
print(f"偏置: {linear_layer.bias.data}")

# 创建输入
x = torch.randn(2, 3)  # batch_size=2, input_dim=3
print(f"\n输入 x: {x}")
print(f"x shape: {x.shape}")

# 前向传播
output = linear_layer(x)
print(f"\n输出: {output}")
print(f"输出shape: {output.shape}")

# 手动计算验证
manual_output = x @ linear_layer.weight.T + linear_layer.bias
print(f"\n手动计算结果: {manual_output}")
print(f"结果相同: {torch.allclose(output, manual_output)}")
```

```python
# 多线性层堆叠
layer1 = nn.Linear(20, 30)
layer2 = nn.Linear(30, 40)
x = torch.randn(128, 20)  # batch_size=128, input_dim=20
print(f"\n输入 x shape: {x.shape}")
# 逐层前向传播
y1 = layer1(x)  # Y1 = XW1
print(f"第一层输出 y1 shape: {y1.shape}")

y2 = layer2(y1)  # Y2 = Y1W2
print(f"第二层输出 y2 shape: {y2.shape}")

# 验证高维情况
x_high_dim = torch.randn(128, 4096, 30, 20)
print(f"\n高维输入 x_high_dim shape: {x_high_dim.shape}")

# nn.Linear会自动处理高维输入，只对最后两个维度进行线性变换
y1_high = layer1(x_high_dim)
print(f"高维输入第一层输出 shape: {y1_high.shape}")

y2_high = layer2(y1_high)
print(f"高维输入第二层输出 shape: {y2_high.shape}")
```

#### 5.反向传播和自动求导

##### (1)反向传播

反向传播是深度学习的关键，其核心思想是通过链式法则计算复合函数的导数。

假设构建模型$f$，其参数为$\theta$，目标函数为$f^*$，想要设计一种可用来度量基于$\theta$的模型预测结果与真实结果差距的度量，差距越小，模型越接近所需估计的函数。

训练模型过程：

优化目标$J(\theta)=\frac{1}{n} \sum_{X\in\chi}(f*(X)-f(X;\theta))^2$

使用梯度下降法求偏导，$f^*(X)$通常以真值的形式体现，因此$\frac{\partial }{\partial \theta}J(\theta)$重点关注$f(X;\theta)$

$f(X)=XW-->\frac{\partial }{\partial \theta}f(X)=\frac{\partial }{\partial \theta}(XW)$

通常深度学习模型为复合函数，可利用链式法则求偏导

核心步骤：针对优化目标$J(\theta)$按层回退，一层一层求偏导。

假设$y=XW=f_3(f_2(f_1(X)))$，由链式法则展开得：

$\frac{\partial J}{\partial {\theta_{f_1}}}=\frac{\partial J}{\partial y}\cdot\frac{\partial y}{\partial f_3}\cdot\frac{\partial f_3}{\partial f_2}\cdot\frac{\partial f_2}{\partial f_1}\cdot\frac{\partial f_1}{\partial {\theta_{f_1}}}$

```python
# 创建需要梯度的参数
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

print(f"输入 x: {x}")
print(f"权重 w: {w}")
print(f"偏置 b: {b}")

# 前向传播: z = x * w + b
z = x * w + b
print(f"\n前向传播结果 z: {z}")

# 定义损失函数 (这里用简单的平方损失)
target = torch.tensor(10.0)
loss = (z - target) ** 2
print(f"目标值: {target}")
print(f"损失值: {loss}")

# 反向传播
loss.backward()

# 查看梯度
print(f"\n梯度信息:")
print(f"∂loss/∂x = {x.grad}")
print(f"∂loss/∂w = {w.grad}")
print(f"∂loss/∂b = {b.grad}")

# 手动验证梯度计算
# loss = (z - target)^2 = (x*w + b - target)^2
# ∂loss/∂x = 2*(x*w + b - target) * w = 2*(z - target) * w
# ∂loss/∂w = 2*(x*w + b - target) * x = 2*(z - target) * x
# ∂loss/∂b = 2*(x*w + b - target) * 1 = 2*(z - target)

manual_grad_x = 2 * (z - target) * w
manual_grad_w = 2 * (z - target) * x
manual_grad_b = 2 * (z - target)

print(f"\n手动计算的梯度:")
print(f"∂loss/∂x = {manual_grad_x}")
print(f"∂loss/∂w = {manual_grad_w}")
print(f"∂loss/∂b = {manual_grad_b}")

print(f"\n梯度计算正确: {torch.allclose(x.grad, manual_grad_x) and torch.allclose(w.grad, manual_grad_w) and torch.allclose(b.grad, manual_grad_b)}")
```

##### (2)计算图：模型计算的DAG图
<P align="center">
    <img src="resources/DAG.png" width="50%">
</p>

例子：

线性层：$z=x\cdot\omega +b$

损失函数：$loss$=CrossEntropy($z$,$y$)

前向传播：输出为$z=x\cdot\omega +b$，输出的grad：$\frac{\partial loss}{\partial z}$

反向传播：

$\frac{\partial loss}{\partial \omega}=\frac{\partial loss}{\partial z}\cdot\frac{\partial z}{\partial \omega}=x\cdot \frac{\partial loss}{\partial z}$

$\frac{\partial loss}{\partial b}=\frac{\partial loss}{\partial z}\cdot\frac{\partial z}{\partial b}=\frac{\partial loss}{\partial z}$

$\frac{\partial loss}{\partial x}=\frac{\partial loss}{\partial z}\cdot\frac{\partial z}{\partial x}=\frac{\partial loss}{\partial z}\cdot \omega^\top$

##### (3)使用Pytorch的核心特性进行自动求导

Pytorch会自动构建DAG图（基于`nn.Module`，`Tensor`,`forward`等），设置Tensore对象的`requires_grad=True`即可启动对该Tensore的梯度计算，使用反向传播方法`backward()`自动计算梯度。

一个简单的深度学习模型实例（autograd的无感知使用）：

```python
# 创建简单的回归模型
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU() # 激活函数，增加非线性能力

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# 创建模型
model = SimpleModel(input_dim=2, hidden_dim=10, output_dim=1)   
print(f"模型: {model}")                                         

# 创建损失函数和优化器
criterion = nn.MSELoss()                                    #均方误差损失
optimizer = optim.SGD(model.parameters(), lr=0.01)          #随机梯度下降

# 生成简单的训练数据 (y = 2*x1 + 3*x2 + 1 + noise)
torch.manual_seed(42)
X_train = torch.randn(100, 2)
y_train = 2 * X_train[:, 0] + 3 * X_train[:, 1] + 1 + 0.1 * torch.randn(100)
y_train = y_train.unsqueeze(1)  # 添加维度

print(f"训练数据 X shape: {X_train.shape}")
print(f"训练数据 y shape: {y_train.shape}")

# 训练循环
num_epochs = 200
losses = []

# 训练模型
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # 反向传播
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数
    
    losses.append(loss.item())
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f"\n最终损失: {losses[-1]:.4f}")

# 测试模型
model.eval()                # 开启评估模式
with torch.no_grad():       # 不计算梯度
    test_input = torch.tensor([[1.0, 2.0]])
    test_output = model(test_input)
    expected = 2 * 1.0 + 3 * 2.0 + 1  # 应该是9
    print(f"\n测试输入: {test_input}")
    print(f"模型预测: {test_output.item():.4f}")
    print(f"期望输出: {expected:.4f}")
    print(f"预测误差: {abs(test_output.item() - expected):.4f}")
```

##### (4)常见反向节点类型 
AddmmBackward: 对应 addmm 的反向（矩阵乘 + 偏置相加），是 `nn.Linear`/`F.linear` 的核心反向(AddBackward或MulBackward)

TBackward: 对应 `transpose` 的反向，是线性层实现里常见的辅助节点（例如 `W.t()`）。反传中将梯度再转回原始维度

AccumulateGrad: 不是算子反向，而是“叶子张量梯度累加”节点。把传来的梯度写入叶子张量（如 `Linear.weight/bias` 或 `requires_grad=True` 的输入）的 `.grad` 中，并按步累加

```python
x = torch.ones(2, 2, requires_grad=True) # 
y = x + 2
z = y * y * 3
out = z.mean()

print(x.grad_fn)  # None，因为x是叶子张量 leaf tensor
print(y.grad_fn)  # <AddBackward0>，表示 y = x + 2
print(z.grad_fn)  # <MulBackward0>，表示 z = y * y * 3
print(out.grad_fn) # <MeanBackward0>，表示 mean 操作
```

##### (5)叶子节点VS非叶子节点
在进行前向传播时进行即时构图，`requires_grad=True`的参与者会被追踪。
- 叶子节点: 叶子是直接由用户创建且需梯度的tensor，其梯度累积在 `.grad`
- 非叶子节点：由运算生成的新tensor，其梯度累计在`.grad_fn`，比如`z=x+y`, z中包含`grad_fn`，是用于记录的Function对象，描述生成该tensor的运算方式，以及反向传播时如何计算梯度

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)       # 叶子
y = 5 * x                                                   # 非叶子
print('x: is_leaf=', x.is_leaf, ', grad_fn=', x.grad_fn)
print('y: is_leaf=', y.is_leaf, ', grad_fn=', y.grad_fn)

# 默认非叶子不保留 .grad，调用 retain_grad() 以便教学观察
y.retain_grad()
loss = y.sum()
loss.backward()
print('x.grad:', x.grad)
print('y.grad (retained):', y.grad)
```

##### (5)反向传播的启动过程
  
当执行`out.backward()`时，PyTorch 会从`out.grad_fn`出发，沿着计算图往回追踪，调用每个`grad_fn`对应的`backward()`方法，逐步计算出各个叶子节点（比如参数`x`）的梯度。

##### (6)非标量的Backward

反向传播需要一个起点，对标量来说，就是1。

对于非标量的情况，例如`y=model(x)`，假设`y.shape=[batch,dim]`，直接调用`y.backward()`会报错，因为torch内部不知道到底该对哪个方向做反向传播，需要指定与y同形的上游向量v:`y.backward(gradient=v)`。

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # 非标量输出 shape=(3,)
try:
    y.backward()
except Exception as e:
    print('直接 backward 报错:', e)

v = torch.ones_like(y) / y.numel() # numel()返回元素个数，模拟均值操作的梯度
print(v)
y.backward(gradient=v)
print('提供 gradient 后 x.grad:', x.grad)
```

##### (7)高阶梯度与`create_graph`

如果需要对一阶梯度再求梯度，需要在第一次求导时设置`create_graph=True`，否则一阶梯度将视作常量，无法继续反向（默认一次反向后释放图）。

```python
x = torch.tensor(3.0, requires_grad=True)
y = x**3
# 一阶梯度：dy/dx = 3x^2
(g1,) = torch.autograd.grad(y, x, create_graph=True)
# 二阶梯度：d^2y/dx^2 = 6x
(g2,) = torch.autograd.grad(g1, x)
print('g1=', float(g1), ' g2=', float(g2))
```

##### (8)多次反向与retain_graph

默认反向后会释放计算图；再次反向需 `retain_graph=True` 或重新前向。

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x**3).sum()
# 第一次 OK
y.backward(retain_graph=True)
print('第一次 x.grad:', x.grad)
# 第二次如不保留图会报错；上面已设置 retain_graph=True，可再次反向
y.backward()
print('第二次 x.grad (累加):', x.grad)
```

##### (9)detach / no_grad / inference_mode 区别

- `x.detach()`: 切断梯度但共享数据存储，常用于停止梯度或缓存。
- `with torch.no_grad()`: 暂停 autograd 记录，常用于推理或 EMA 更新参数；
- `with torch.inference_mode()`: 进一步优化推理内存与速度（不可写视角）。

#### 6.Broadcasting广播机制

许多PyTorch操作支持类似Numpy的广播机制，允许不同维度的张量在运行算自动进行维度的扩展。

广播规则总结：
- 从右向左对齐维度
- 维度必须兼容：相等、其中一个为1或其中一个不存在
- 不兼容的维度会自动扩展

##### (1)标量与tensor的广播

```python
a = torch.tensor([1, 2, 3, 4])
b = 2
c = a + b
# 手动方法
d = a + torch.full_like(a, b)
print(f"a + b = {c} 形状:a={a.shape}, c={c.shape}")
print(torch.allclose(c, d))
```

##### (2)不同形状tensor的广播

```python
A = torch.randn(3, 4)
B = torch.ones(4)
print(f"A形状: {A.shape}")
print(f"B形状: {B.shape}")

print("A:", A)
print("B:", B)
# 广播过程：B [4] -> [1, 4] -> [3, 4]
C = A + B
# 手动方法
D = A + B.unsqueeze(0).expand(3, 4)
print(f"A + B形状: {C.shape}")
print("C:", C)
print(f"广播是否成功: {C.shape == (3, 4)}")
print(torch.allclose(C, D))
```

##### (3)复杂情况下的广播

```python
A = torch.randn(2, 3, 4)
B = torch.randn(3, 1) # 对齐大小相同的维度
print(A)
print(B)
result_broadcast = A + B
print(result_broadcast.shape)
print(result_broadcast)
```

##### (4)矩阵乘法中的广播

```python
A = torch.randn(3, 4)
B = torch.randn(2, 4, 5)
result_broadcast = torch.matmul(A, B)
# 等价于
A_expanded = A.unsqueeze(0).expand(2, 3, 4) #(3, 4)->(1, 3, 4)->(2, 3, 4)
result_manual = torch.bmm(A_expanded, B)

print(result_broadcast.shape)
```

##### (5)批量归一化中的广播

```python
x = torch.randn(32, 64, 28, 28)

mean_broadcast = torch.mean(x, dim=(0, 2, 3), keepdim=True)
std_broadcast = torch.std(x, dim=(0, 2, 3), keepdim=True)
normalized_broadcast = (x - mean_broadcast) / (std_broadcast + 1e-8)

print(normalized_broadcast.shape)
```

##### (6)注意力机制中的广播

```python
scores = torch.randn(2, 8, 8)
# print(scores)
mask = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
masked_scores_broadcast = scores + mask.unsqueeze(0).unsqueeze(0)
# print(masked_scores_broadcast)
print(masked_scores_broadcast.shape)
```

#### 7.转置操作

`transpose()`侧重于交换两个维度：`def transpose(self, dim0: _int, dim1: _int) -> Tensor`。

而`permute()`可以实现多个维度的重新排列：`def permute(self, dims: _size) -> Tensor: `，其中dims为代表交换顺序的数组。

```python
x = torch.randn(2, 3, 4, 5)
print(f"原始tensor形状: {x.shape}")
# 使用transpoe交换维度
y1 = x.transpose(1, 3)
print(f"transpose(1, 3)后形状: {y1.shape}")
print(f"原始tensor[0, 1, 2, 3] = {x[0, 1, 2, 3]}")
print(f"转置后tensor[0, 3, 2, 1] = {y1[0, 3, 2, 1]}")
print(f"两者是否相等: {x[0, 1, 2, 3] == y1[0, 3, 2, 1]}")
```

```python
# 使用permute重排维度
y2 = x.permute(0, 3, 1, 2)
print(f"permute(0, 3, 1, 2)后形状: {y2.shape}")
print(f"原始tensor[0, 1, 2, 3] = {x[0, 1, 2, 3]}")
print(f"permute后tensor[0, 3, 1, 2] = {y2[0, 3, 1, 2]}")
print(f"两者是否相等: {x[0, 1, 2, 3] == y2[0, 3, 1, 2]}")
```

转置操作导致内存上不连续：转置时并没有真的搬运内存中的数据，只是修改了“如何读取数据”的规则（元数据），具体表现为，逻辑上相邻的元素，在物理内存中不再相邻。

```python
# 连续性问题
print(f"原始tensor是否连续: {x.is_contiguous()}")
print(f"transpose后是否连续: {y1.is_contiguous()}")
print(f"permute后是否连续: {y2.is_contiguous()}")
```

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
# 线性内存 |1|2|3|4|5|6|
y = torch.transpose(x, 0, 1)
# 内存上仍然是 |1|2|3|4|5|6|，但是读取[1, 4]的4时跳跃内存
```

#### 8.形状变换操作：view vs reshape

`view()`与`reshape()`的主要区别在于能否处理不连续tensor。

```python
x = torch.randn(2, 3, 4)
print(x.is_contiguous())
# view操作
y1 = x.view(6, 4)
print(y1.is_contiguous())
# reshape操作
y2 = x.reshape(6, 4)
print(y2.is_contiguous())
```

```python
# view()处理不连续tensor会报错
x_transposed = x.transpose(0, 1)
print(f"转置后是否连续: {x_transposed.is_contiguous()}")
try:
    y3 = x_transposed.view(6, 4)
except RuntimeError as e:
    print("view报错信息:", e)
```

```python
# reshape() 会自动处理连续性问题
y4 = x_transposed.reshape(12, 2)
print(f"reshape()成功: {y4.shape}")
```

```python
# 验证是否共享内存，即是否会随着原始tensor修改数据而改变
x[0, 0, 0] = 1
print(f"修改原始tensor后，view结果: {y1[0, 0]}")
print(f"修改原始tensor后，reshape结果: {y2[0, 0]}")
x[0, 0, 0] = 999
print(f"修改原始tensor后，view结果: {y1[0, 0]}")
print(f"修改原始tensor后，reshape结果: {y2[0, 0]}")
```

#### 9.维度操作

使用`squeeze`进行某个维度的移除，使用`unsqueeze`在指定位置插入大小为1的维度。

```python
x = torch.randn(1, 3, 1, 4)
print(x.shape)
# 移除所有大小为1的维度
y1 = x.squeeze()
print(y1.shape)
# 移除指定维度
y2 = x.squeeze(0)
print(y2.shape)
```

```python
x = torch.randn(3, 4)
print(x.shape)
y1 = x.unsqueeze(0)
print(y1.shape)
y2 = x.unsqueeze(-1)
print(y2.shape)
y3 = x.unsqueeze(1)
print(y3.shape)
```

使用`torch.flatten()`来合并某个维度

torch.flatten(input, start_dim=0, end_dim=-1) → Tensor

```python
t = torch.tensor([[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]])
print(t.shape)
print(torch.flatten(t))
print(torch.flatten(t).shape)
print(torch.flatten(t, start_dim=1))
print(torch.flatten(t, start_dim=1).shape)
```

#### 10.索引、分散与选择

`gather()`用来按照索引收集元素并创建新tensor，遵循以下规则（以一个三维tensor为例）：

        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

```python
x = torch.randn(3, 4)
print(x)
# 索引的维度需要和目标tensor匹配，除了dim的维度
indices = torch.tensor([0, 2, 1])
# 在第1维上按索引收集元素
y1 = torch.gather(x, 1, indices.unsqueeze(1))
print(y1)
print(y1.shape)
# 手动验证
print(f"第0行选择第{indices[0]}列: {x[0, indices[0]]}, 结果: {y1[0, 0]}")
print(f"第1行选择第{indices[1]}列: {x[1, indices[1]]}, 结果: {y1[1, 0]}")
print(f"第2行选择第{indices[2]}列: {x[2, indices[2]]}, 结果: {y1[2, 0]}")
```

`scatter_()`函数按照索引来将目标tensor中的值替换为某个tensor中的值

```python
x_scatter = torch.zeros(3, 4)
values = torch.randn(3, 2)
indices_scatter = torch.tensor([[0, 2], [1, 3], [0, 1]])
print(f"目标tensor:\n{x_scatter}")
print(f"要分散的值:\n{values}")
print(f"分散索引:\n{indices_scatter}")
# 分散元素
x_scatter.scatter_(1, indices_scatter, values)
print(f"scatter后结果:\n{x_scatter}")
```

`index_select()`函数可以按照索引选择并提取整行/整列元素

```python
# 按索引选择列
x = torch.randn(5, 6)
print(x)
indices = torch.tensor([0, 2, 4])
selected = torch.index_select(x, 1, indices)
print(f"原始tensor形状: {x.shape}")
print(f"选择的索引: {indices}")
print(f"选择结果形状: {selected.shape}")
print(f"选择结果:\n{selected}")
```

`masked_select()`函数按照掩码选择并保留元素，结果为一维tensor

```python
# 创建掩码
x = torch.randn(3, 4)
mask = x > 0.5
print(f"原始tensor:\n{x}")

# 按掩码选择
selected_values = torch.masked_select(x, mask)
print(f"按掩码选择的值: {selected_values}")
print(f"选择的值数量: {len(selected_values)}")
```

#### 11.插入元素

```python
x = torch.randn(1, 2)
print(x)
y1 = torch.stack((x, x)) # 等价于 torch.stack((x, x), dim=0)
print("y1:",y1)
print(y1.shape)
y2 = torch.stack((x, x), dim=1)
print("y2:",y2)
print(y2.shape)
y3 = torch.stack((x, x), dim=2)
print("y3:",y3)
print(y3.shape)
y4 = torch.stack((x, x), dim=-1)
print("y4:",y4)
print(y4.shape)
```

#### 12.外积

```python
v1 = torch.arange(1., 5.)
print(v1)
v2 = torch.arange(1., 4.)
print(v2)
print(torch.outer(v1, v2))
```
