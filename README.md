# NeuMatC: 神经矩阵方程求解器

基于神经网络学习矩阵逆的连续映射，实现矩阵方程的快速求解。

## 核心思想

将矩阵方程求解问题转换为**参数化连续映射问题**：
- 传统方法：对每个矩阵单独求解 `AX=B`
- NeuMatC：学习参数 `p → A(p)⁻¹` 的连续映射，一次训练解决整族矩阵

---

## 数学原理

### 1. 参数化矩阵族

使用低秩分解构造参数化矩阵：
```
H(p) = A(p) @ B(p)ᵀ + εI
```
其中 `A(p)`, `B(p)` 通过正弦/余弦函数参数化，确保 `H(p)` 光滑连续。

### 2. 学习目标

直接学习矩阵的逆：
```
AX=B  →  学习 A⁻¹
XA=B  →  学习 A⁻¹
```

Ground Truth 使用双精度求解保证准确性：
```python
A_inv = torch.linalg.solve(H.double(), torch.eye(n).double()).float()
```

### 3. 低秩连续映射模型

```python
X(p) = C ×₃ Φ(p)
```
- `Φ(p)`: MLP(p)  →  潜在向量
- `C`: 可学习张量 n×n×latent_dim
- `×₃`: mode-3 张量-矩阵乘法

### 4. 两阶段训练

**Phase 1: 纯监督预训练**
```python
loss = ||pred - gt||_F²
lr = 1e-4, 1000轮
```

**Phase 2: 一致性损失微调**
```python
data_loss = ||pred - gt||_F²
consist_loss = ||H @ pred - I||_F² / (n*n)
loss = data_loss + λ * consist_loss
lr = 1e-5
```

### 5. 矩阵条件数

条件数 κ(A) = ||A||·||A⁻¹|| 衡量病态程度：
- κ < 100: 良态
- 100 < κ < 1e5: 中等病态
- κ > 1e5: 严重病态

---

## 项目结构

```
src/
├── models/
│   └── low_rank_continuous_mapping.py  # 低秩连续映射模型
├── solver/
│   ├── trainer.py                       # 两阶段训练逻辑
│   └── algebraic_solver.py             # 代数损失与求解
└── utils/
    ├── data_generator.py                # 参数化矩阵生成
    ├── adaptive_sampling.py            # 自适应采样
    └── matrix_error_handler.py         # 矩阵错误处理
main.py                                  # 主入口
```

---

## 使用方法

### 环境要求

```bash
pip install torch numpy
```

### 快速开始

```bash
python main.py
```

### 配置参数

修改 [main.py](main.py) 中的超参数：

```python
# 超参数
MATRIX_SIZE = 256      # 矩阵维度 n×n
OP_TYPE = 'inv'        # 'inv' 或 'svd'
EQUATION_TYPE = 'inv'  # 'inv', 'AX=B', 'XA=B'
```

### 训练流程

1. **Phase 1 (纯监督预训练, 1000轮)**
   - 学习率: 1e-4
   - 仅使用 data_loss = ||pred - gt||_F²
   - 收敛后保存至 `phase1_inverse.pth`

2. **Phase 2 (一致性约束微调, 5000轮)**
   - 学习率: 1e-5
   - 损失: data_loss + λ × consist_loss
   - λ 自适应调整，平衡两个目标

### 自定义训练

```python
from src.solver.trainer import train_neumatc, test_model, baseline_test

# 训练模型
model = train_neumatc(
    n=256,              # 矩阵大小
    op='inv',           # 操作类型
    equation_type='AX=B',  # 方程类型
    num_train=40,       # 训练样本数
    num_test=100,       # 测试样本数
    max_iter=5000,      # Phase 2 最大迭代
    update_T=500        # 自适应采样间隔
)

# 测试模型
test_model(model, n=256, op='inv', equation_type='AX=B')

# NumPy 基线对比
baseline_test(n=256, op='inv')
```

### 测试指标

| 方程类型 | 指标 | 含义 |
|---------|------|------|
| `inv` | \|\|H @ H⁻¹ - I\|\| / \|\|I\|\| | 逆矩阵精度 |
| `AX=B` | \|\|H @ (H⁻¹ @ B) - B\|\| / \|\|B\|\| | 方程求解精度 |
| `XA=B` | \|\|(H⁻¹ @ B) @ H - B\|\| / \|\|B\|\| | 方程求解精度 |

---

## 测试结果

**配置**: n=256, num_test=100, 参数分布: [0,0.1]∪[0.9,1]∪[0.1,0.9]

| 方程类型 | 相对误差 | 推理时间 |
|---------|----------|----------|
| `inv`   | 2.15e-02 | 11.46 ms |
| `AX=B`  | 2.14e-02 | 10.24 ms |

Phase 1 后误差约 2%，需 Phase 2 进一步降低。

---

## 核心代码

### 推理求解 (trainer.py)

```python
# 给定参数 p 和矩阵 H(p)
p_tensor = torch.tensor([[p]], dtype=torch.float32)
H = generate_parametric_matrix(p, n)
pred_inv = model(p_tensor)  # 预测的逆矩阵

# 求解 AX=B
B = torch.randn(n, n)
X_pred = torch.mm(pred_inv.squeeze(), B)
err = torch.norm(torch.mm(H, X_pred) - B) / torch.norm(B)
```

### 模型定义 (low_rank_continuous_mapping.py)

```python
model = LowRankContinuousMapping(
    input_dim=1,      # 参数 p 的维度
    hidden_dim=256,   # MLP 隐藏层维度
    latent_dim=128,   # 潜在空间维度
    output_shape=(n, n),  # 输出矩阵维度
    activation='sin'  # 激活函数
)
```
