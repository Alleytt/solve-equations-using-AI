"""
NeuMatC 消融实验脚本
用于验证两阶段训练、动态自适应损失平衡、主动纠错式自适应采样三大创新点的独立贡献

实验配置：
- Base: 仅保留基础训练框架，作为性能基线
- Abl-1: 引入自适应损失平衡策略，验证其对收敛性的增益
- Abl-2: 随机增加监督点，作为对照验证自适应采样的价值
- Abl-3: 仅失败引导采样，不加监督补充，验证纠错机制必要性
- Full: 高残差点补充值 + 监督训练，完整验证系统性能

矩阵规模: 128 × 128
训练样本: 初始40点 → 补至200点
训练迭代: P1:500次 | P2:2500次
优化策略: P1:SGD | P2:Adam (lr=1e-5)
评估指标: 50个测试点误差中位数 + 标准差(5种子)
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
from datetime import datetime

torch.manual_seed(42)
np.random.seed(42)

# ================== 模型定义 ==================
class LowRankContinuousMapping(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, latent_dim=128, output_shape=(128, 128)):
        super().__init__()
        self.output_shape = output_shape
        self.latent_dim = latent_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.C = nn.Parameter(torch.randn(output_shape[0], output_shape[1], latent_dim) * 0.1)
    
    def forward(self, p):
        phi = self.mlp(p)
        B = phi.shape[0]
        n1, n2, d = self.C.shape
        C_flat = self.C.reshape(n1 * n2, d)
        out = torch.mm(phi, C_flat.t())
        return out.reshape(B, n1, n2)

# ================== 参数化矩阵生成 ==================
def generate_parametric_matrix(p, n=128):
    """生成可控条件数的参数化矩阵"""
    r = 20
    A0_raw = np.random.randn(n, r)
    B0_raw = np.random.randn(n, r)
    A0 = A0_raw / np.linalg.norm(A0_raw, axis=0, keepdims=True)
    B0 = B0_raw / np.linalg.norm(B0_raw, axis=0, keepdims=True)
    
    FA = np.random.uniform(0.5, 1.5, r)
    FB = np.random.uniform(0.5, 1.5, r)
    PhiA = np.random.uniform(0, 2*np.pi, r)
    PhiB = np.random.uniform(0, 2*np.pi, r)
    
    Asin = A0 * np.sin(2 * np.pi * FA * p + PhiA)
    Bcos = B0 * np.cos(2 * np.pi * FB * p + PhiB)
    A = Asin @ Bcos.T + 0.05 * np.eye(n)
    return torch.from_numpy(A.astype(np.float32))

def get_ground_truth(p, n, op='inv'):
    """获取逆矩阵的ground truth"""
    H = generate_parametric_matrix(p, n)
    I = torch.eye(n)
    return torch.linalg.solve(H.double(), I.double()).float()

# ================== 自适应采样策略 ==================
def failure_guided_sampling(model, n, num_candidates=1000, num_select=10):
    """失败引导采样：选择残差最大的点"""
    p_candidates = np.random.uniform(0, 1, num_candidates)
    residuals = []
    
    model.eval()
    with torch.no_grad():
        for p in p_candidates:
            H = generate_parametric_matrix(p, n).unsqueeze(0)
            pred = model(torch.tensor([[p]], dtype=torch.float32))
            residual = torch.bmm(H, pred) - torch.eye(n).unsqueeze(0)
            res_norm = torch.norm(residual, p='fro').item()
            residuals.append((p, res_norm))
    
    # 选择残差最大的点
    residuals.sort(key=lambda x: -x[1])
    selected = [p for p, _ in residuals[:num_select]]
    return selected

def random_sampling(num_select=10):
    """随机采样"""
    return np.random.uniform(0, 1, num_select).tolist()

# ================== 训练函数 ==================
def train_experiment(config_name, n=128, num_train=40, max_iter_p1=500, max_iter_p2=2500,
                     use_dynamic_lambda=True, sampling_strategy='none', add_supervised=True,
                     seed=42):
    """
    训练单个消融实验配置
    
    Args:
        config_name: 配置名称 (Base, Abl-1, Abl-2, Abl-3, Full)
        n: 矩阵大小
        num_train: 初始训练样本数
        use_dynamic_lambda: 是否使用动态损失平衡
        sampling_strategy: 采样策略 ('none', 'random', 'fgs')
        add_supervised: 是否添加监督训练
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = LowRankContinuousMapping(output_shape=(n, n), hidden_dim=256, latent_dim=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 生成初始训练数据
    p_train = np.random.uniform(0, 1, num_train).tolist()
    train_data = []
    for p in p_train:
        H = generate_parametric_matrix(p, n).to(device)
        gt = get_ground_truth(p, n).to(device)
        train_data.append((torch.tensor([[p]], dtype=torch.float32).to(device), 
                          H.unsqueeze(0), gt.unsqueeze(0)))
    
    I = torch.eye(n, device=device).unsqueeze(0)
    total_samples = 200  # 目标总样本数
    
    # Phase 1: 纯监督预训练
    print(f"[{config_name}] Phase 1: 纯监督预训练")
    optimizer_p1 = torch.optim.SGD(model.parameters(), lr=1e-3)
    for it in range(max_iter_p1):
        model.train()
        optimizer_p1.zero_grad()
        total_loss = 0.0
        
        for p_tensor, H_p, gt in train_data:
            pred = model(p_tensor)
            loss = torch.norm(pred - gt, p='fro')**2
            loss.backward()
            total_loss += loss.item()
        
        optimizer_p1.step()
        
        if it % 100 == 0:
            print(f"  Iter {it}: loss={total_loss/len(train_data):.4f}")
    
    # Phase 2: 一致性约束微调
    print(f"[{config_name}] Phase 2: 一致性约束微调")
    optimizer_p2 = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    
    avg_data_loss = None
    avg_consist_loss = None
    update_interval = 500
    target_samples = min(total_samples, len(train_data) + 5 * 32)  # 最多添加5轮
    
    for it in range(max_iter_p2):
        model.train()
        optimizer_p2.zero_grad()
        total_loss = 0.0
        total_data_loss = 0.0
        total_consist_loss = 0.0
        
        for p_tensor, H_p, gt in train_data:
            pred = model(p_tensor)
            
            # 数据损失
            data_loss = torch.norm(pred - gt, p='fro')**2
            
            # 一致性损失
            residual = torch.bmm(H_p, pred) - I
            consist_loss = torch.norm(residual, p='fro')**2 / (n * n)
            
            # 更新移动平均
            if avg_data_loss is None:
                avg_data_loss = data_loss.item()
                avg_consist_loss = consist_loss.item()
            else:
                avg_data_loss = 0.9 * avg_data_loss + 0.1 * data_loss.item()
                avg_consist_loss = 0.9 * avg_consist_loss + 0.1 * consist_loss.item()
            
            # 动态lambda
            if use_dynamic_lambda:
                lambda_consist = avg_data_loss / (avg_consist_loss + 1e-8)
            else:
                lambda_consist = 0.1  # 固定lambda
            
            # 组合损失
            if add_supervised:
                loss = data_loss + lambda_consist * consist_loss
            else:
                loss = lambda_consist * consist_loss
            
            loss.backward()
            total_loss += loss.item()
            total_data_loss += data_loss.item()
            total_consist_loss += consist_loss.item()
        
        optimizer_p2.step()
        
        # 自适应采样
        if it % update_interval == 0 and it > 0 and len(train_data) < target_samples:
            if sampling_strategy == 'fgs':
                new_points = failure_guided_sampling(model, n, num_select=min(32, target_samples - len(train_data)))
            elif sampling_strategy == 'random':
                new_points = random_sampling(num_select=min(32, target_samples - len(train_data)))
            else:
                new_points = []
            
            # 添加新训练点
            for p in new_points:
                H = generate_parametric_matrix(p, n).to(device)
                gt = get_ground_truth(p, n).to(device)
                train_data.append((torch.tensor([[p]], dtype=torch.float32).to(device), 
                                  H.unsqueeze(0), gt.unsqueeze(0)))
            
            if len(new_points) > 0:
                print(f"  Iter {it}: 添加 {len(new_points)} 个训练点, 总数={len(train_data)}")
        
        if it % 500 == 0:
            print(f"  Iter {it}: total_loss={total_loss/len(train_data):.4f}, "
                  f"data_loss={total_data_loss/len(train_data):.4f}, "
                  f"consist_loss={total_consist_loss/len(train_data):.4f}")
    
    return model

# ================== 测试函数 ==================
def test_model(model, n=128, num_test=50):
    model.eval()
    device = next(model.parameters()).device
    errors = []
    
    p_test = np.random.uniform(0, 1, num_test)
    
    with torch.no_grad():
        for p in p_test:
            H = generate_parametric_matrix(p, n).to(device)
            pred = model(torch.tensor([[p]], dtype=torch.float32).to(device))
            I = torch.eye(n, device=device)
            err = torch.norm(H @ pred.squeeze() - I) / torch.norm(I)
            errors.append(err.item())
    
    return np.mean(errors), np.std(errors), np.median(errors)

# ================== 消融实验配置 ==================
def run_ablation_study():
    """运行完整的消融实验"""
    configs = {
        'Base': {
            'use_dynamic_lambda': False,
            'sampling_strategy': 'none',
            'add_supervised': True
        },
        'Abl-1': {
            'use_dynamic_lambda': True,
            'sampling_strategy': 'none',
            'add_supervised': True
        },
        'Abl-2': {
            'use_dynamic_lambda': True,
            'sampling_strategy': 'random',
            'add_supervised': True
        },
        'Abl-3': {
            'use_dynamic_lambda': True,
            'sampling_strategy': 'fgs',
            'add_supervised': False  # 仅失败引导采样，不加监督补充
        },
        'Full': {
            'use_dynamic_lambda': True,
            'sampling_strategy': 'fgs',
            'add_supervised': True
        }
    }
    
    results = {}
    num_seeds = 5
    
    print("="*70)
    print("NeuMatC 消融实验")
    print("="*70)
    print(f"矩阵规模: 128×128")
    print(f"训练样本: 初始40点 → 补至200点")
    print(f"训练迭代: P1={500}次 | P2={2500}次")
    print(f"评估指标: {50}个测试点的误差 (5种子)")
    print("="*70)
    
    for config_name, config_params in configs.items():
        print(f"\n--- 实验配置: {config_name} ---")
        print(f"  动态λ: {config_params['use_dynamic_lambda']}")
        print(f"  采样策略: {config_params['sampling_strategy']}")
        print(f"  监督训练: {config_params['add_supervised']}")
        
        all_means = []
        all_stds = []
        all_medians = []
        
        for seed in range(num_seeds):
            print(f"\n  种子 {seed+1}/{num_seeds}")
            start_time = time.time()
            
            model = train_experiment(
                config_name,
                n=128,
                num_train=40,
                max_iter_p1=500,
                max_iter_p2=2500,
                seed=seed,
                **config_params
            )
            
            mean_err, std_err, median_err = test_model(model, n=128, num_test=50)
            elapsed = time.time() - start_time
            
            all_means.append(mean_err)
            all_stds.append(std_err)
            all_medians.append(median_err)
            
            print(f"  测试结果: mean={mean_err:.4e}, std={std_err:.4e}, median={median_err:.4e}")
            print(f"  耗时: {elapsed:.2f}秒")
        
        # 汇总结果
        results[config_name] = {
            'mean_error': np.mean(all_means),
            'std_error': np.mean(all_stds),
            'median_error': np.median(all_medians),
            'all_means': all_means,
            'all_medians': all_medians,
            'params': config_params
        }
        
        print(f"\n  === {config_name} 汇总 ===")
        print(f"  平均误差: {results[config_name]['mean_error']:.4e}")
        print(f"  标准差: {results[config_name]['std_error']:.4e}")
        print(f"  中位数误差: {results[config_name]['median_error']:.4e}")
    
    return results

# ================== 结果保存与分析 ==================
def save_results(results):
    """保存消融实验结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = 'results/ablation'
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存JSON结果
    json_path = os.path.join(save_dir, f'ablation_results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 生成Markdown报告
    md_path = os.path.join(save_dir, f'ablation_summary_{timestamp}.md')
    with open(md_path, 'w') as f:
        f.write(f"# NeuMatC 消融实验报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 实验设置\n\n")
        f.write(f"- 矩阵规模: 128 × 128\n")
        f.write(f"- 训练样本: 初始40点 → 补至200点\n")
        f.write(f"- 训练迭代: Phase 1={500}次 | Phase 2={2500}次\n")
        f.write(f"- 评估指标: 50个测试点误差 (5种子)\n\n")
        
        f.write(f"## 实验配置\n\n")
        f.write(f"| 配置 | 两阶段训练 | 动态λ | 采样策略 | 监督训练 |\n")
        f.write(f"|------|------------|-------|----------|----------|\n")
        for config_name, data in results.items():
            params = data['params']
            f.write(f"| {config_name} | ✔️ | {'✔️' if params['use_dynamic_lambda'] else '❌'} | "
                    f"{params['sampling_strategy']} | {'✔️' if params['add_supervised'] else '❌'} |\n")
        
        f.write(f"\n## 实验结果\n\n")
        f.write(f"| 配置 | 平均误差 | 标准差 | 中位数误差 |\n")
        f.write(f"|------|----------|--------|------------|\n")
        for config_name, data in results.items():
            f.write(f"| {config_name} | {data['mean_error']:.4e} | {data['std_error']:.4e} | {data['median_error']:.4e} |\n")
        
        # 计算相对提升
        base_error = results['Base']['median_error']
        f.write(f"\n## 相对性能提升 (相对于Base)\n\n")
        f.write(f"| 配置 | 中位数误差 | 相对提升 |\n")
        f.write(f"|------|------------|----------|\n")
        for config_name, data in results.items():
            improvement = (1 - data['median_error'] / base_error) * 100
            f.write(f"| {config_name} | {data['median_error']:.4e} | {improvement:.1f}% |\n")
    
    print(f"\n结果已保存到: {save_dir}")
    return json_path, md_path

if __name__ == '__main__':
    results = run_ablation_study()
    json_path, md_path = save_results(results)
    
    print("\n" + "="*70)
    print("消融实验完成！")
    print("="*70)
    print(f"JSON结果: {json_path}")
    print(f"报告文件: {md_path}")
    print("="*70)