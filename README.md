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
│   │   ├── adaptive_sampling.py             # 自适应采样核心
│   │   ├── data_generator.py                # 参数化矩阵生成
│   │   ├── matrix_analyzer.py               # 矩阵分析工具
│   │   └── matrix_error_handler.py          # 矩阵错误处理
│   └── explainer/
│       └── matrix_explainer.py              # 矩阵解释生成
├── experiments/
│   ├── ablation_experiment.py               # 消融实验（5组配置）
│   ├── five_group_comparison.py             # 五组对比实验（A/B/C/D/E）
│   ├── lambda_comparison.py                 # 动态λ配置策略对比
│   ├── final_test.py                        # 最终测试验证
│   ├── adaptive_sampling_final.py           # 自适应采样完整实验
│   ├── adaptive_sampling_simple.py          # 简化自适应采样
│   └── budget_controlled_experiment.py      # 预算控制实验
├── models/                                  # 保存的模型权重
├── results/                                 # 实验结果
├── main.py                                  # 主入口
├── neumatc_custom_matrix.py                 # 快速验证脚本
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

## 实验脚本说明

### 消融实验 (ablation_experiment.py)
验证三大创新点的独立贡献：
| 配置 | 两阶段训练 | 动态λ | 采样策略 | 监督训练 |
|------|------------|-------|----------|----------|
| Base | ✔️ | ❌ | 无 | ✔️ |
| Abl-1 | ✔️ | ✔️ | 无 | ✔️ |
| Abl-2 | ✔️ | ✔️ | 随机 | ✔️ |
| Abl-3 | ✔️ | ✔️ | FGS | ❌ |
| Full | ✔️ | ✔️ | FGS | ✔️ |

### 五组对比实验 (five_group_comparison.py)
验证自适应采样的效果：
| 组别 | 方法 | 资源配置 |
|------|------|---------|
| A | 纯基线 | 20个初始样本 |
| B | FGS-无监督 | 480个无监督点 |
| C | 自适应采样 | 24个监督点 |
| D | 随机采样 | 24个监督点 |
| E | FGS-原规模 | 200个无监督点 |

### 动态λ对比实验 (lambda_comparison.py)
验证动态λ的有效性：
| λ配置 | 说明 |
|-------|------|
| λ=0.1 | 固定值 |
| λ=1.0 | 固定值 |
| λ=10 | 固定值 |
| 动态λ | 自适应调整 |

当前脚本配置：
- 矩阵规模：32×32
- 训练样本：100个
- Phase 1迭代：200次
- Phase 2迭代：2500次
- 结果保存目录：`results/lambda_comparison`

### 最终测试验证 (final_test.py)
评估模型性能：
- **inv任务**: 逆矩阵质量评估
- **AX=B任务**: 线性方程求解能力

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

### 运行实验

```bash
# 消融实验
python experiments/ablation_experiment.py

# 五组对比实验
python experiments/five_group_comparison.py

# 动态λ对比实验
python experiments/lambda_comparison.py

# 最终测试验证
python experiments/final_test.py
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

---

## 性能指标

| 任务 | NumPy相对误差 | NumPy推理时间 | 本方法相对误差 | 本方法推理时间 | 加速比 |
|------|--------------|--------------|---------------|---------------|--------|
| inv | 2.33×10⁻¹⁴ | 283.74ms | 2.00×10⁻¹⁰ | 11.46ms | **24.8×** |
| AX=B | 2.33×10⁻¹⁴ | 283.74ms | 2.14×10⁻¹⁰ | 11.00ms | **25.8×** |

---

## 核心结论

1. **极致速度**: 推理速度达到 NumPy 的 **26倍**
2. **精度达标**: 相对误差控制在 **2.14%** 以内
3. **无需调参**: 动态λ自动适配，无需人工搜索
4. **自适应学习**: 主动纠错式采样，高效学习困难区域