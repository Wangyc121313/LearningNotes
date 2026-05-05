# Machine Learning (机器学习)

> 经典/传统机器学习算法学习笔记

## 内容概览

- 线性模型（Linear Regression、Logistic Regression）
- 支持向量机（SVM）
- 决策树与集成方法（Decision Tree、Random Forest、XGBoost）
- 聚类算法（K-Means、DBSCAN）
- 降维（PCA、t-SNE）
- 模型评估与调参

## 1. 线性模型

### 线性回归（Linear Regression）
$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b, \quad L_{MSE} = \frac{1}{N}\sum(y_i - \hat{y}_i)^2$$
- 闭合解：$\mathbf{w} = (X^\top X)^{-1} X^\top \mathbf{y}$（小数据集）
- Ridge（L2 正则）：防止过拟合；Lasso（L1）：产生稀疏解

### 逻辑回归（Logistic Regression）
输出概率而非连续值：$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b)$，损失为交叉熵。
> 本质是线性分类器，决策边界为超平面。

---

## 2. 支持向量机（SVM）

寻找**最大间隔超平面**：$\min \frac{1}{2}\|\mathbf{w}\|^2$，s.t. $y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1$

- **硬间隔**：数据线性可分（无噪声）
- **软间隔**：允许少量误分类（引入松弛变量 + 惩罚 C）
- **核技巧**（Kernel Trick）：通过核函数 $K(\mathbf{x}_i, \mathbf{x}_j)$ 在高维空间中线性分割非线性数据

| 核函数 | 公式 | 适用 |
|--------|------|------|
| Linear | $\mathbf{x}_i^\top \mathbf{x}_j$ | 线性可分 |
| RBF/Gaussian | $\exp(-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2)$ | 非线性，通用 |
| Polynomial | $(\mathbf{x}_i^\top \mathbf{x}_j + c)^d$ | 多项式边界 |

---

## 3. 决策树与集成方法

### 决策树
- 递归选择**最优分裂特征**（最大化信息增益 / 最小化基尼系数）
- 剪枝防止过拟合（限制最大深度、最小样本数等）

### Bagging（Bootstrap Aggregating）
并行训练多棵树，各自独立，**投票/平均**。

**Random Forest** = Bagging + 每次分裂随机选 $\sqrt{d}$ 个特征 → 降低相关性，更鲁棒

### Boosting（串行，逐步纠错）
| 方法 | 原理 |
|------|------|
| **AdaBoost** | 迭代提高误分类样本权重 |
| **Gradient Boosting** | 每棵树拟合前一轮残差（梯度方向） |
| **XGBoost** | GBDT + 正则化 + 二阶导数 + 并行，工业标准 |
| **LightGBM** | 按梯度采样 + 叶分裂策略，速度更快 |

---

## 4. 聚类算法

### K-Means
1. 随机初始化 k 个质心
2. 每个样本分配到最近质心
3. 重新计算质心，重复至收敛

- 超参数：k（用肘部法则选取）
- 局限：假设球形簇，对噪声敏感，需预定 k

### DBSCAN（Density-Based）
- **核心点**：邻域内至少 MinPts 个点
- **边界点**：在核心点邻域内
- **噪声点**：既不是核心点也不是边界点

优点：可发现任意形状簇，自动识别噪声。无需预定 k。

---

## 5. 降维

### PCA（主成分分析）
找方差最大的正交方向（主成分），保留前 k 个成分。

1. 数据中心化：$X \leftarrow X - \bar{X}$
2. 计算协方差矩阵：$C = \frac{1}{N}X^\top X$
3. 特征分解：$C = V\Lambda V^\top$，取前 k 列

### t-SNE
非线性降维，保留局部邻域结构，主要用于**可视化**（不用于特征工程）。

---

## 6. 模型评估

### 分类指标
| 指标 | 公式 | 适用 |
|------|------|------|
| Accuracy | $\frac{TP+TN}{All}$ | 类别均衡 |
| Precision | $\frac{TP}{TP+FP}$ | 关注假阳性 |
| Recall | $\frac{TP}{TP+FN}$ | 关注漏检 |
| F1 | $\frac{2 \cdot P \cdot R}{P+R}$ | 类别不均衡 |
| AUC-ROC | TPR vs FPR 曲线下面积 | 二分类综合评估 |

### 交叉验证
- **K-Fold**：将数据分 k 份，轮流作验证集，结果更可靠
- **Stratified K-Fold**：保持各折类别比例一致（类别不均衡时用）

### 超参数调优
```python
# GridSearchCV：穷举搜索
# RandomizedSearchCV：随机搜索（大搜索空间）
# Optuna：贝叶斯优化（更智能）
```

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs, make_circles
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ===== 数据生成 =====
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                            n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# ===== 1. 线性模型 =====
for name, clf in [
    ("LogisticRegression", LogisticRegression(C=1.0, max_iter=1000)),
    ("Ridge Classifier", LogisticRegression(C=0.1, penalty='l2')),   # l2 正则
]:
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"{name}: AUC={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ===== 2. SVM =====
svm_pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True))])
svm_pipe.fit(X_train, y_train)
y_prob = svm_pipe.predict_proba(X_test)[:, 1]
print(f"\nSVM (RBF kernel): Test AUC={roc_auc_score(y_test, y_prob):.4f}")

# ===== 3. 集成方法 =====
models = {
    "Decision Tree":      DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":      RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "Gradient Boosting":  GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
}
print("\n--- 集成方法对比 ---")
for name, clf in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    pipe.fit(X_train, y_train)
    auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    print(f"{name:25s}: AUC={auc:.4f}")

# Random Forest 特征重要性
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
top5_idx = np.argsort(rf.feature_importances_)[::-1][:5]
print(f"\nRandom Forest Top-5 重要特征: {top5_idx}")
print(f"对应重要性: {rf.feature_importances_[top5_idx].round(3)}")

# ===== 4. 聚类：K-Means =====
X_blob, y_blob = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

# 肘部法则选 k
inertias = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_blob)
    inertias.append(km.inertia_)
print(f"\nK-Means Inertia (k=1..9): {[f'{v:.0f}' for v in inertias]}")

km_best = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = km_best.fit_predict(X_blob)
print(f"K=4 聚类完成，标签种类: {np.unique(labels)}")

# DBSCAN
X_circ, _ = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
db = DBSCAN(eps=0.3, min_samples=5)
db_labels = db.fit_predict(X_circ)
n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = np.sum(db_labels == -1)
print(f"\nDBSCAN: 发现 {n_clusters} 个簇, {n_noise} 个噪声点")

# ===== 5. PCA 降维 =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA: 20维 → 2维")
print(f"解释方差比: {pca.explained_variance_ratio_}")
print(f"累计解释方差: {pca.explained_variance_ratio_.sum():.2%}")

# 保留 95% 方差所需主成分数
pca_95 = PCA(n_components=0.95)
X_95 = pca_95.fit_transform(X_scaled)
print(f"保留95%方差需要 {pca_95.n_components_} 个主成分")

# ===== 6. 模型评估报告 =====
best_clf = RandomForestClassifier(n_estimators=100, random_state=42)
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)
print("\n=== 分类报告 ===")
print(classification_report(y_test, y_pred))
```
