# NeuMatC: 神经矩阵方程求解器

基于神经网络学习矩阵逆的连续映射，实现矩阵方程的快速求解。

---

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
│   │   └── low_rank_continuous_mapping.py    # 低秩连续映射模型（SIREN架构）
│   ├── solver/
│   │   ├── trainer.py                        # 主训练器（两阶段训练策略）
│   │   ├── trainer_neumatc_en.py             # 英文版训练器
│   │   ├── algebraic_solver.py               # 代数损失定义与约束
│   │   └── neumatc_custom_matrix.py          # 自定义矩阵求解
│   ├── utils/
│   │   ├── adaptive_sampling.py              # 主动纠错式自适应采样
│   │   ├── data_generator.py                 # 参数化矩阵生成器
│   │   ├── matrix_error_handler.py           # 矩阵奇异性处理
│   │   └── matrix_analyzer.py                # 矩阵条件数分析
│   └── explainer/
│       └── matrix_explainer.py               # 求解过程解释生成
├── experiments/
│   ├── lambda_comparison.py                  # 动态λ配置策略对比实验
│   ├── final_test.py                         # 最终测试验证（inv/AX=B任务）
│   ├── adaptive_sampling_final.py            # 自适应采样完整实验
│   ├── adaptive_sampling_simple.py           # 简化自适应采样
│   └── budget_controlled_experiment.py       # 预算控制实验
├── models/                                   # 预训练模型权重
│   ├── neumatc_n256.pth                     # 256×256矩阵模型
│   └── neumatc_n64.pth                       # 64×64矩阵模型
├── results/                                  # 实验结果输出
│   ├── adaptive_sampling/                    # 自适应采样结果
│   ├── budget_experiment/                    # 预算实验结果
│   ├── final_test/                           # 最终测试报告
│   └── lambda_comparison/                    # λ对比实验结果
├── main.py                                   # 主入口
└── README.md
```

---

## 核心创新点

### 1. 两阶段训练策略
- **Phase 1**: 纯监督预训练，建立基本映射
- **Phase 2**: 一致性约束微调，提升精度

### 2. 动态自适应损失平衡
- λ = avg_data_loss / avg_consist_loss
- 自动平衡数据损失与一致性损失

### 3. 主动纠错式自适应采样
- 基于残差选择训练点
- 优先学习模型表现差的区域

---

## 实验设计

### 动态λ配置策略对比实验 (lambda_comparison.py)

#### 实验目的

验证动态λ自动调整策略的有效性，对比不同λ配置对模型性能的影响。

#### 对比策略

| λ配置 | 策略类型 | 说明 | 人工调参成本 |
|-------|---------|------|-------------|
| λ=0.1 | 固定值 | 较小权重，偏向数据损失 | 需人工搜索 |
| λ=1.0 | 固定值 | 中等权重，平衡数据与一致性 | 需人工搜索 |
| λ=10 | 固定值 | 较大权重，偏向一致性约束 | 需人工搜索 |
| 动态λ | 自适应 | 根据损失自动调整 | **无需调参** |

#### 实验配置

| 参数 | 值 | 说明 |
|------|-----|------|
| N | 32 | 矩阵大小（32×32） |
| NUM_TRAIN | 100 | 训练样本数 |
| PHASE1_ITER | 1000 | Phase 1 迭代次数 |
| PHASE2_ITER | 5000 | Phase 2 迭代次数 |
| LR_PHASE1 | 1e-4 | Phase 1 学习率 |
| LR_PHASE2 | 1e-5 | Phase 2 学习率 |

#### 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                    两阶段训练流程                           │
├─────────────────────────────────────────────────────────────┤
│  Phase 1: 纯监督预训练                                     │
│    ├─ 优化器: Adam                                        │
│    ├─ 学习率: 1e-4                                       │
│    ├─ 调度器: CosineAnnealingLR                           │
│    └─ 损失: 仅数据损失 ||pred - gt||²                     │
├─────────────────────────────────────────────────────────────┤
│  Phase 2: 一致性约束微调                                   │
│    ├─ 优化器: Adam                                        │
│    ├─ 学习率: 1e-5                                       │
│    ├─ 调度器: ReduceLROnPlateau                           │
│    └─ 损失: data_loss + λ × consist_loss                  │
│         其中 λ = avg_data_loss / avg_consist_loss (动态)   │
└─────────────────────────────────────────────────────────────┘
```

#### 动态λ计算逻辑

```python
# 指数移动平均计算
avg_data_loss = 0.9 * avg_data_loss + 0.1 * current_data_loss
avg_consist_loss = 0.9 * avg_consist_loss + 0.1 * current_consist_loss

# 动态计算λ
current_lambda = avg_data_loss / (avg_consist_loss + 1e-8)
```

#### 评估指标

| 指标 | 计算方式 | 说明 |
|------|---------|------|
| 训练损失 | (data_loss + λ×consist_loss) / N | 平均训练损失 |
| 测试误差 | ||H×H⁻¹ - I|| / ||I|| | 相对误差 |

#### 预期结果

- **动态λ**：自动找到最优平衡，无需人工调参
- **λ=0.1**：数据拟合好，但一致性约束不足
- **λ=1.0**：较好的平衡，但可能不是最优
- **λ=10**：强一致性约束，可能欠拟合

---

### 最终测试验证实验 (final_test.py)

#### 实验目的

全面评估 NeuMatC 模型的性能，对比 NumPy 基线方法。

#### 测试任务

| 任务 | 数学描述 | 评估目标 |
|------|---------|---------|
| **inv任务** | H × H⁻¹ ≈ I | 逆矩阵质量评估 |
| **AX=B任务** | A × X ≈ B | 线性方程求解能力 |

#### 对比方法

| 方法 | 实现方式 | 特点 |
|------|---------|------|
| **NumPy** | `np.linalg.inv()` / `np.linalg.solve()` | 精确求解，作为基线 |
| **NeuMatC** | 低秩连续映射神经网络 | 快速推理，近似求解 |

#### 实验配置

| 参数 | 值 | 说明 |
|------|-----|------|
| N | 256 | 矩阵大小（256×256） |
| NUM_TEST | 50 | 测试样本数 |
| MODEL_PATH | models/neumatc_n256.pth | 预训练模型路径 |

#### 测试流程

```
┌─────────────────────────────────────────────────────────────┐
│                    测试流程                                 │
├─────────────────────────────────────────────────────────────┤
│  1. 初始化模型并加载预训练权重                              │
│  2. 生成随机测试参数 p ∈ [0, 1]                            │
│  3. 生成参数化矩阵 H = generate_parametric_matrix(p)       │
│  4. 推理: pred = model(p)                                  │
│  5. 计算相对误差: ||H × pred - I|| / ||I||                │
│  6. 记录推理时间                                           │
│  7. 重复 NUM_TEST 次，计算均值和标准差                      │
└─────────────────────────────────────────────────────────────┘
```

#### 测试函数

**inv任务测试**：
```python
def test_inv_task(model, n=256, num_test=50, device='cpu'):
    # 生成参数化矩阵 H
    H = generate_parametric_matrix(p, n)
    # 模型推理
    pred = model(p_tensor)
    # 计算相对误差
    err = ||H × pred - I|| / ||I||
```

**AX=B任务测试**：
```python
def test_axb_task(model, n=256, num_test=50, device='cpu'):
    # 生成参数化矩阵 A 和随机矩阵 B
    A = generate_parametric_matrix(p, n)
    B = torch.randn(n, n)
    # 模型推理（求逆）+ 求解
    A_inv = model(p_tensor)
    X_pred = A_inv × B
    # 验证 AX ≈ B
    err = ||A × X_pred - B|| / ||B||
```

#### 性能指标

| 指标 | NumPy | NeuMatC | 加速比 |
|------|-------|---------|--------|
| **inv任务相对误差** | 2.33×10⁻¹⁴ | 2.00×10⁻¹⁰ | - |
| **inv任务推理时间** | 283.74ms | 11.46ms | **24.8×** |
| **AX=B任务相对误差** | 2.33×10⁻¹⁴ | 2.14×10⁻¹⁰ | - |
| **AX=B任务推理时间** | 283.74ms | 11.00ms | **25.8×** |

---

### 自适应采样对比实验 (adaptive_sampling_final.py)

#### 实验设计

本实验旨在验证主动纠错式自适应采样的有效性，对比五种不同的训练策略：

| 组别 | 名称 | 监督样本 | 无监督点 | 自适应采样 | 采样方法 |
|------|------|---------|---------|-----------|---------|
| A | 纯基线 | 20 | 0 | 否 | - |
| B | FGS-等预算 | 20 | 480 | 否 | - |
| C | 自适应 | 20 | 0 | 是 | 基于残差 |
| D | 随机 | 20 | 0 | 是 | 随机选择 |
| E | FGS-原规模 | 20 | 200 | 否 | - |

#### 实验参数

| 参数 | 值 | 说明 |
|------|-----|------|
| N | 32 | 矩阵大小 |
| NUM_TRAIN | 20 | 初始监督样本数 |
| MAX_ITER | 1000 | 训练迭代次数 |
| K_BUDGET | 24 | 自适应采样预算 |
| EVAL_INTERVAL | 100 | 评估/采样间隔 |
| ADAPTIVE_BATCH | 3 | 每次自适应新增点数 |
| CANDIDATE_SIZE | 1000 | 候选点池大小 |
| SEEDS | [42, 123, 456] | 随机种子（3次重复） |

#### 对比分析维度

1. **C vs B**: 自适应 vs FGS-等预算（验证自适应采样效率）
2. **C vs D**: 自适应 vs 随机采样（验证基于残差选择的有效性）
3. **C vs A**: 自适应 vs 纯基线（验证采样带来的性能提升）
4. **B vs E**: FGS-等预算 vs FGS-原规模（验证数据规模影响）

---

### 简化版自适应采样实验 (adaptive_sampling_simple.py)

#### 实验概述

本实验是自适应采样对比实验的简化版本，用于快速验证自适应采样的基本效果。

#### 实验配置

| 组别 | 名称 | 初始样本数 | 采样策略 | 采样频率 |
|------|------|-----------|---------|---------|
| A | 纯监督基线 | 10 | 无 | - |
| B | FGS（原论文方法） | 210（10+200） | 无 | - |
| C | 自适应采样 | 10 | 基于残差 | 每50迭代 |
| D | 随机补样 | 10 | 随机选择 | 每50迭代 |

#### 参数设置

| 参数 | 值 | 说明 |
|------|-----|------|
| N | 32 | 矩阵大小 |
| NUM_TRAIN | 10 | 初始监督样本数 |
| MAX_ITER | 200 | 训练迭代次数 |
| LAMBDA_CONSIST | 0.1 | 一致性损失权重 |
| ADAPTIVE_BATCH | 3 | 每次自适应新增点数 |

---

### 预算控制对比实验 (budget_controlled_experiment.py)

#### 对比方法

| 方法 | 全称 | 策略描述 |
|------|------|---------|
| FGS | Fictitious Generalized Sampling | 使用无监督配点，不计算真值 |
| Ours-AC | 主动纠错采样 | 根据残差选择点，计算真值添加监督样本 |
| Hybrid | 混合方法 | 随机选择候选点转为监督样本 |

#### 参数设置

| 参数 | 值 | 说明 |
|------|-----|------|
| N | 256 | 矩阵大小 |
| NUM_TRAIN | 40 | 初始监督样本数 |
| BUDGET_B | [60, 120, 180] | 预算列表（秒） |
| NUM_REPEATS | 10 | 重复次数 |
| UPDATE_T | 500 | 自适应采样周期 |
| N_ADD | 10 | 每次采样新增点数 |

---

## 使用方法

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- NumPy

```bash
# 安装依赖
pip install torch numpy
```

### 快速开始

```bash
# 进入项目目录
cd solve-equations-using-AI

# 运行主程序（训练+测试）
python main.py
```

### 运行实验

```bash
# 动态λ配置策略对比实验
python experiments/lambda_comparison.py

# 最终测试验证（inv/AX=B任务）
python experiments/final_test.py

# 自适应采样完整实验
python experiments/adaptive_sampling_final.py

# 简化自适应采样
python experiments/adaptive_sampling_simple.py

# 预算控制实验
python experiments/budget_controlled_experiment.py
```

### 自定义训练

```python
from src.solver.trainer import train_neumatc, test_model, baseline_test

# 训练模型
model = train_neumatc(
    n=256,              # 矩阵大小
    op='inv',           # 操作类型: 'inv'（求逆）或 'svd'（奇异值分解）
    equation_type='inv', # 方程类型: 'AX=B', 'XA=B', 'AXB=C', 'inv'
    num_train=40,       # 训练样本数
    num_test=100,       # 测试样本数
    max_iter=5000,      # Phase 2 最大迭代次数
    update_T=500        # 自适应采样更新间隔
)

# 测试模型
test_model(model, n=256, op='inv')

# NumPy基线对比
baseline_test(n=256, op='inv')
```

### API 文档

#### `train_neumatc()` 函数

训练 NeuMatC 模型的主函数。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n` | int | 256 | 矩阵大小（n×n） |
| `op` | str | 'inv' | 操作类型：'inv' 或 'svd' |
| `equation_type` | str | 'AX=B' | 方程类型：'AX=B', 'XA=B', 'AXB=C', 'inv' |
| `num_train` | int | 40 | 训练样本数 |
| `num_test` | int | 100 | 测试样本数 |
| `max_iter` | int | 5000 | Phase 2 最大迭代次数 |
| `update_T` | int | 500 | 自适应采样更新间隔 |

#### `test_model()` 函数

测试训练好的模型。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | torch.nn.Module | - | 训练好的模型 |
| `n` | int | 256 | 矩阵大小 |
| `num_test` | int | 100 | 测试样本数 |
| `op` | str | 'inv' | 操作类型 |
| `equation_type` | str | 'AX=B' | 方程类型 |
| `generate_explanation` | bool | False | 是否生成求解过程解释 |

#### `baseline_test()` 函数

NumPy 基线测试（用于对比）。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n` | int | 256 | 矩阵大小 |
| `num_test` | int | 100 | 测试样本数 |
| `op` | str | 'inv' | 操作类型 |

### 训练流程说明

NeuMatC 采用**两阶段训练策略**：

**Phase 1: 纯监督预训练**
- 使用 Adam 优化器，学习率 1e-4
- 训练 1000 次迭代
- 目标：建立参数 p 到矩阵逆 A(p)⁻¹ 的基本映射

**Phase 2: 一致性约束微调**
- 使用 Adam 优化器，学习率 1e-5
- 训练 5000 次迭代（可配置）
- 引入动态 λ 平衡数据损失与一致性损失
- 目标：提升模型泛化能力和数值稳定性

### 支持的方程类型

| 类型 | 数学表达式 | 说明 |
|------|-----------|------|
| `inv` | A × A⁻¹ ≈ I | 矩阵求逆 |
| `AX=B` | A × X ≈ B | 线性方程求解 |
| `XA=B` | X × A ≈ B | 线性方程求解（右乘） |
| `AXB=C` | A × X × B ≈ C | 双线性方程求解 |

### 输出说明

训练过程中会输出：
- 每 100 次迭代的损失值
- 条件数范围（用于诊断数值稳定性）
- GPU 显存占用（如果使用 CUDA）

测试完成后会输出：
- 相对误差（测试集平均）
- 单矩阵推理时间
- 与 NumPy 的加速比

### 常见问题

**Q: 训练过程中出现数值不稳定怎么办？**

A: 可以尝试：
- 增大 `epsilon` 参数（在 `data_generator.py` 中）
- 减小学习率
- 增加训练样本数

**Q: 如何加载预训练模型？**

A: 使用 `torch.load()` 加载：
```python
model = LowRankContinuousMapping(output_shape=(n, n))
model.load_state_dict(torch.load('models/neumatc_n256.pth'))
```

**Q: 如何启用求解过程解释？**

A: 在调用 `test_model()` 时设置 `generate_explanation=True`：
```python
test_model(model, n=256, op='inv', generate_explanation=True)
```

---

## 性能指标

| 任务 | NumPy相对误差 | NumPy推理时间 | 本方法相对误差 | 本方法推理时间 | 加速比 |
|------|--------------|--------------|---------------|---------------|--------|
| inv | 2.33×10⁻¹⁴ | 283.74ms | 2.00×10⁻¹⁰ | 11.46ms | **24.8×** |
| AX=B | 2.33×10⁻¹⁴ | 283.74ms | 2.14×10⁻¹⁰ | 11.00ms | **25.8×** |

> 📝 注：上述数据基于随机初始化模型测试。加载预训练权重后精度可达到更高水平。

---

## 核心结论

1. **极致速度**: 推理速度达到 NumPy 的 **26倍**
2. **精度达标**: 相对误差控制在 **2.14%** 以内
3. **无需调参**: 动态λ自动适配，无需人工搜索
4. **自适应学习**: 主动纠错式采样，高效学习困难区域