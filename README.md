# NeuMatC: 神经矩阵方程求解器

基于神经网络学习矩阵逆的连续映射，实现矩阵方程的快速求解。

## 核心思想

将矩阵方程求解问题转换为**参数化连续映射问题**：
- 传统方法：对每个矩阵单独求解 `AX=B`
- NeuMatC：学习参数 `p → A(p)⁻¹` 的连续映射，一次训练解决整族矩阵

---

## 项目结构

```
solve-equations-using-AI/
├── src/
│   ├── models/
│   │   └── low_rank_continuous_mapping.py    # 低秩连续映射模型
│   ├── solver/
│   │   ├── trainer.py                       # 主训练器（含两阶段训练）
│   │   ├── trainer_neumatc_en.py           # 英文版两阶段训练
│   │   └── algebraic_solver.py              # 代数损失定义
│   ├── utils/
│   │   ├── adaptive_sampling.py             # 自适应采样
│   │   ├── data_generator.py                # 参数化矩阵生成
│   │   ├── matrix_analyzer.py               # 矩阵分析工具
│   │   └── matrix_error_handler.py          # 矩阵错误处理
│   └── explainer/
│       └── matrix_explainer.py              # 矩阵解释生成
├── experiments/
│   ├── adaptive_sampling_final.py           # 自适应采样完整实验
│   ├── adaptive_sampling_simple.py          # 简化自适应采样
│   └── budget_controlled_experiment.py      # 预算控制实验
├── models/                                  # 保存的模型权重
├── results/                                 # 实验结果
├── main.py                                  # 主入口
└── README.md
```

---

## 核心模块

### 1. 模型 (low_rank_continuous_mapping.py)

低秩连续映射网络，核心思想是将输出矩阵分解为：

```python
X(p) = C ×₃ Φ(p)
```

- `Φ(p) = MLP(p)` → 潜在向量
- `C` → 可学习张量 n×n×latent_dim
- `×₃` → mode-3 张量-矩阵乘法

```python
model = LowRankContinuousMapping(
    input_dim=1,        # 参数 p 的维度
    hidden_dim=256,     # MLP 隐藏层维度
    latent_dim=128,     # 潜在空间维度
    output_shape=(n, n), # 输出矩阵维度
    activation='sin'
)
```

### 2. 两阶段训练 (trainer.py)

**Phase 1: 纯监督预训练**
```
loss = ||pred - gt||_F²
学习率: 1e-4
迭代: 1000 轮
```
- 使用少量标注数据学习逆矩阵的基本映射
- Ground Truth 通过双精度求解保证准确性

**Phase 2: 一致性约束微调**
```
data_loss = ||pred - gt||_F²
consist_loss = ||H @ pred - I||_F² / (n*n)
loss = data_loss + λ × consist_loss
学习率: 1e-5
迭代: 5000 轮
```
- 一致性损失强制满足 H @ H⁻¹ ≈ I 约束
- λ 自适应调整：λ = avg_data_loss / (avg_consist_loss + 1e-8)

### 3. 自适应采样 (adaptive_sampling.py)

在 Phase 2 过程中，根据当前模型在参数空间中的表现动态选择新的采样点：

```python
def failure_informed_sampling(model, A_func, candidate_p,
                               epsilon_r=1e-3,   # 残差阈值
                               epsilon_p=0.05,   # 失败概率阈值
                               N_add=10):        # 每次添加点数
```

- 计算候选点的残差：||H(p) @ pred(p) - I||
- 选择残差最大的点加入训练集
- 优先学习模型表现差的区域

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

### 自定义训练

```python
from src.solver.trainer import train_neumatc, test_model, baseline_test

model = train_neumatc(
    n=256,              # 矩阵大小
    op='inv',           # 操作类型: 'inv' 或 'svd'
    equation_type='AX=B', # 方程类型: 'AX=B', 'XA=B', 'inv'
    num_train=40,       # 训练样本数
    num_test=100,       # 测试样本数
    max_iter=5000,      # Phase 2 最大迭代
    update_T=500        # 自适应采样间隔
)

test_model(model, n=256, op='inv')
baseline_test(n=256, op='inv')
```

### 配置参数 (main.py)

```python
MATRIX_SIZE = 256      # 矩阵维度 n×n
OP_TYPE = 'inv'        # 'inv' 或 'svd'
EQUATION_TYPE = 'inv'  # 'inv', 'AX=B', 'XA=B'
```

---

## 训练流程详解

### Phase 1: 纯监督预训练 (1000轮)

```
训练数据: 40个随机采样点 (p ∈ [0,1])
损失函数: ||pred(p) - H(p)⁻¹||_F²
学习率: 1e-4 (Adam)
```

**目标**：建立逆矩阵映射的基本近似

### Phase 2: 一致性约束微调 (5000轮)

```
损失函数: data_loss + λ × consist_loss
    data_loss = ||pred - gt||_F²
    consist_loss = ||H @ pred - I||_F² / (n*n)
学习率: 1e-5 (Adam)
自适应采样: 每500轮更新候选点
```

**目标**：通过一致性约束提升精度

### 自适应采样流程

```
每500轮:
1. 对1000个候选点计算残差
2. 选择残差最大的10个点加入训练集
3. 优先学习模型表现差的区域
```

---

## 测试指标

| 指标 | 公式 | 含义 |
|------|------|------|
| 相对误差 | \|\|H @ H⁻¹ - I\|\| / \|\|I\|\| | 逆矩阵精度 |
| 推理时间 | ms/matrix | 单矩阵求解耗时 |

---

## 数学背景

### 参数化矩阵族

使用低秩分解构造光滑的矩阵族：
```
H(p) = A(p) @ B(p)ᵀ + εI
A(p)ᵢⱼ = sin(2πfᵢp + φᵢ)
B(p)ᵢⱼ = cos(2πgᵢp + ψᵢ)
```

### 条件数

条件数 κ(A) = ||A||·||A⁻¹|| 衡量数值稳定性：
- κ < 100: 良态矩阵
- 100 < κ < 1e5: 中等病态
- κ > 1e5: 严重病态

---

## 推理示例

```python
import torch
from src.models.low_rank_continuous_mapping import LowRankContinuousMapping
from src.utils.data_generator import generate_parametric_matrix

model = LowRankContinuousMapping(input_dim=1, hidden_dim=256,
                                  latent_dim=128, output_shape=(256, 256))
model.load_state_dict(torch.load('results/Base_model.pth'))

p = 0.5
p_tensor = torch.tensor([[p]], dtype=torch.float32)
H = generate_parametric_matrix(p, 256)

pred_inv = model(p_tensor).squeeze()
I = torch.eye(256)
error = torch.norm(H @ pred_inv - I) / torch.norm(I)
print(f"相对误差: {error:.4e}")
```
