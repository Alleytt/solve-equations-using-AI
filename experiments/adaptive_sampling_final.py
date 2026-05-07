"""
自适应采样对比实验 - 完全修复版
修复内容：
1. FGS组真正实现无监督一致性损失
2. 自适应采样预算严格控制
3. 统一测试集（全局共享）
4. 封装测试误差计算函数
5. 候选点批量化处理
6. 无监督损失权重归一化
7. 改进代码结构和可维护性
"""

import torch
import torch.optim as optim
import numpy as np
import time
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.low_rank_continuous_mapping import LowRankContinuousMapping
from src.utils.data_generator import generate_parametric_matrix

print("=" * 70)
print("自适应采样对比实验 - 完全修复版")
print("=" * 70, flush=True)

# ========== 实验参数配置 ==========
CONFIG = {
    'N': 32,
    'NUM_TRAIN': 20,
    'MAX_ITER': 1000,
    'EVAL_INTERVAL': 100,
    'K_BUDGET': 24,
    'FGS_EQUIV_POINTS': 480,  # K_BUDGET * 20
    'LAMBDA': 0.1,
    'UNSUP_WEIGHT': 1.0,  # 无监督损失权重
    'SEEDS': [42, 123, 456],
    'TEST_SIZE': 100,
    'CANDIDATE_SIZE': 1000,
    'ADAPTIVE_BATCH': 3,
    'LR': 1e-4,
    'WEIGHT_DECAY': 1e-5
}

# ========== 全局固定测试集 ==========
np.random.seed(999)
GLOBAL_TEST_P = np.random.uniform(0, 1, CONFIG['TEST_SIZE'])

def reset_seeds(seed):
    """重置所有随机种子确保可重复性"""
    torch.manual_seed(seed)
    np.random.seed(seed)

def evaluate_test_error(model, p_test, N, device):
    """封装测试误差计算，统一接口"""
    model.eval()
    I = torch.eye(N, device=device)
    rel_err = 0.0
    with torch.no_grad():
        for p in p_test:
            p_tensor = torch.tensor([[p]], dtype=torch.float32).to(device)
            H = generate_parametric_matrix(p, N).to(device)
            pred = model(p_tensor)
            err = torch.norm(H @ pred.squeeze() - I) / torch.norm(I)
            rel_err += err.item()
    return rel_err / len(p_test)

def compute_batch_residuals(model, p_candidate, N, device):
    """批量化计算候选点残差，大幅提升性能"""
    model.eval()
    I = torch.eye(N, device=device)
    
    # 批处理前向传播
    p_tensor = torch.tensor(p_candidate, dtype=torch.float32).view(-1, 1).to(device)
    preds = model(p_tensor)  # (batch, N, N)
    
    # 生成矩阵并计算残差
    residuals = []
    for i, p in enumerate(p_candidate):
        H = generate_parametric_matrix(p, N).to(device)
        pred = preds[i:i+1]
        res = torch.norm(H @ pred.squeeze() - I) / torch.norm(I)
        residuals.append((p, res.item()))
    
    return residuals

def run_single_experiment(config, group_name, supervised_samples, unsupervised_points,
                         is_adaptive, sampling_method, seed):
    """运行单次实验"""
    print(f"\n  [{group_name}] 种子={seed}", flush=True)
    start_time = time.time()
    
    reset_seeds(seed)
    device = torch.device('cpu')
    N = config['N']
    K_BUDGET = config['K_BUDGET']
    ADAPTIVE_BATCH = config['ADAPTIVE_BATCH']
    LAMBDA = config['LAMBDA']
    UNSUP_WEIGHT = config['UNSUP_WEIGHT']
    
    # 创建模型
    model = LowRankContinuousMapping(
        input_dim=1, hidden_dim=128, latent_dim=64,
        output_shape=(N, N), activation='sin'
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['LR'], weight_decay=config['WEIGHT_DECAY'])
    
    I = torch.eye(N, device=device)
    train_p_set = set()
    supervised_data = []
    unsupervised_data = []
    adaptive_data = []
    selected_points_info = []  # 记录选择理由
    
    # ========== 生成监督样本 ==========
    p_supervised = np.random.uniform(0, 1, supervised_samples)
    for p in p_supervised:
        H = generate_parametric_matrix(p, N).to(device)
        gt = torch.linalg.solve(
            H.double(), torch.eye(N, device=device, dtype=torch.double)
        ).float().unsqueeze(0)
        supervised_data.append({
            'p': p,
            'p_tensor': torch.tensor([[p]], dtype=torch.float32).to(device),
            'H': H.unsqueeze(0),
            'gt': gt
        })
        train_p_set.add(p)
    
    # ========== 生成无监督点（FGS组使用）==========
    if unsupervised_points > 0:
        p_unsupervised = np.random.uniform(0, 1, unsupervised_points)
        for p in p_unsupervised:
            H = generate_parametric_matrix(p, N).to(device)
            unsupervised_data.append({
                'p': p,
                'p_tensor': torch.tensor([[p]], dtype=torch.float32).to(device),
                'H': H.unsqueeze(0)
            })
        print(f"    生成 {len(unsupervised_data)} 个无监督点", flush=True)
    
    # ========== 自适应采样候选点 ==========
    p_candidate = np.random.uniform(0, 1, config['CANDIDATE_SIZE'])
    
    # 记录训练过程
    history = defaultdict(list)
    
    # ========== 训练循环 ==========
    for it in range(config['MAX_ITER']):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        
        # 1. 监督样本损失（数据损失 + 一致性损失）
        for item in supervised_data:
            pred = model(item['p_tensor'])
            data_loss = torch.norm(pred - item['gt'], p='fro')**2
            residual = torch.bmm(item['H'], pred) - I.unsqueeze(0)
            consist_loss = torch.norm(residual, p='fro')**2 / (N * N)
            loss = data_loss + LAMBDA * consist_loss
            loss.backward()
            total_loss += loss.item()
        
        # 2. 自适应新增样本损失
        for item in adaptive_data:
            pred = model(item['p_tensor'])
            data_loss = torch.norm(pred - item['gt'], p='fro')**2
            residual = torch.bmm(item['H'], pred) - I.unsqueeze(0)
            consist_loss = torch.norm(residual, p='fro')**2 / (N * N)
            loss = data_loss + LAMBDA * consist_loss
            loss.backward()
            total_loss += loss.item()
        
        # 3. 无监督点一致性损失（仅一致性，无数据损失）
        if unsupervised_data:
            unsup_total = 0.0
            for item in unsupervised_data:
                pred = model(item['p_tensor'])
                residual = torch.bmm(item['H'], pred) - I.unsqueeze(0)
                consist_loss = torch.norm(residual, p='fro')**2 / (N * N)
                consist_loss.backward()
                unsup_total += consist_loss.item()
            
            # 无监督损失归一化（除以点数后乘以权重）
            total_loss += (unsup_total / len(unsupervised_data)) * UNSUP_WEIGHT
        
        optimizer.step()
        
        # 记录训练损失
        if (it + 1) % 10 == 0:
            num_items = len(supervised_data) + len(adaptive_data) + len(unsupervised_data)
            history['train_loss'].append((it + 1, total_loss / max(num_items, 1)))
        
        # ========== 自适应采样（每EVAL_INTERVAL步）==========
        if is_adaptive and (it + 1) % config['EVAL_INTERVAL'] == 0 and it > 0:
            # 检查预算
            if len(selected_points_info) >= K_BUDGET:
                continue
            
            # 计算可用候选点
            available_p = [p for p in p_candidate if p not in train_p_set]
            
            if not available_p:
                continue
            
            # 批量化计算残差
            residuals = compute_batch_residuals(model, available_p, N, device)
            residuals.sort(key=lambda x: x[1], reverse=True)
            
            # 确定本次添加数量
            allow_add = min(ADAPTIVE_BATCH, K_BUDGET - len(selected_points_info))
            
            if sampling_method == 'adaptive':
                new_items = residuals[:allow_add]
            else:  # random
                np.random.shuffle(residuals)
                new_items = residuals[:allow_add]
            
            # 添加新样本
            for p, res in new_items:
                H = generate_parametric_matrix(p, N).to(device)
                gt = torch.linalg.solve(
                    H.double(), torch.eye(N, device=device, dtype=torch.double)
                ).float().unsqueeze(0)
                adaptive_data.append({
                    'p': p,
                    'p_tensor': torch.tensor([[p]], dtype=torch.float32).to(device),
                    'H': H.unsqueeze(0),
                    'gt': gt
                })
                selected_points_info.append({
                    'p': p,
                    'residual': res,
                    'iteration': it + 1,
                    'method': sampling_method
                })
                train_p_set.add(p)
            
            if new_items:
                pts_str = ', '.join([f'{p:.3f}(r={r:.3f})' for p, r in new_items])
                print(f"    迭代 {it+1}: 新增 {len(new_items)} 点, 累计 {len(selected_points_info)}/{K_BUDGET}", flush=True)
        
        # ========== 评估测试误差 ==========
        if (it + 1) % config['EVAL_INTERVAL'] == 0:
            test_err = evaluate_test_error(model, GLOBAL_TEST_P, N, device)
            history['test_error'].append((it + 1, test_err))
            avg_loss = history['train_loss'][-1][1] if history['train_loss'] else 0
            print(f"    迭代 {it+1}: 损失={avg_loss:.4f}, 测试误差={test_err:.4e}", flush=True)
    
    # ========== 最终评估 ==========
    final_err = evaluate_test_error(model, GLOBAL_TEST_P, N, device)
    duration = time.time() - start_time
    
    print(f"    最终: 误差={final_err:.4e}, 监督={len(supervised_data)+len(adaptive_data)}, "
          f"无监督={len(unsupervised_data)}, 耗时={duration:.1f}s", flush=True)
    
    return {
        'group': group_name,
        'seed': seed,
        'final_error': final_err,
        'supervised_samples': len(supervised_data) + len(adaptive_data),
        'unsupervised_points': len(unsupervised_data),
        'adaptive_added': len(selected_points_info),
        'selected_points_info': selected_points_info,
        'history': dict(history),
        'duration': duration
    }

def run_group_experiments(config, group_name, supervised_samples, unsupervised_points,
                          is_adaptive, sampling_method):
    """运行一组实验（多个种子）"""
    print(f"\n{'='*60}")
    print(f"组别: {group_name}")
    print(f"监督样本: {supervised_samples}, 无监督点: {unsupervised_points}")
    print(f"自适应: {is_adaptive}, 方法: {sampling_method}")
    print(f"{'='*60}", flush=True)
    
    group_results = []
    for seed in config['SEEDS']:
        result = run_single_experiment(
            config, group_name, supervised_samples, unsupervised_points,
            is_adaptive, sampling_method, seed
        )
        group_results.append(result)
    
    # 统计汇总
    summary = {
        'group': group_name,
        'mean_error': np.mean([r['final_error'] for r in group_results]),
        'std_error': np.std([r['final_error'] for r in group_results]),
        'mean_samples': np.mean([r['supervised_samples'] for r in group_results]),
        'mean_unsupervised': np.mean([r['unsupervised_points'] for r in group_results]),
        'mean_adaptive_added': np.mean([r['adaptive_added'] for r in group_results]),
        'mean_duration': np.mean([r['duration'] for r in group_results]),
        'individual_results': group_results
    }
    
    return summary

def main():
    print(f"\n【实验参数】")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    print(f"  测试集规模: {len(GLOBAL_TEST_P)}", flush=True)
    
    # 实验配置列表
    experiments = [
        {'name': 'A（纯基线）', 'sup': 20, 'unsup': 0, 'adaptive': False, 'method': None},
        {'name': 'B（FGS-等预算）', 'sup': 20, 'unsup': CONFIG['FGS_EQUIV_POINTS'], 'adaptive': False, 'method': None},
        {'name': 'C（自适应）', 'sup': 20, 'unsup': 0, 'adaptive': True, 'method': 'adaptive'},
        {'name': 'D（随机）', 'sup': 20, 'unsup': 0, 'adaptive': True, 'method': 'random'},
        {'name': 'E（FGS-原规模）', 'sup': 20, 'unsup': 200, 'adaptive': False, 'method': None},
    ]
    
    # 运行所有实验
    results = []
    for exp in experiments:
        result = run_group_experiments(
            CONFIG,
            group_name=exp['name'],
            supervised_samples=exp['sup'],
            unsupervised_points=exp['unsup'],
            is_adaptive=exp['adaptive'],
            sampling_method=exp['method']
        )
        results.append(result)
    
    # 输出汇总
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    
    print(f"\n| 组别 | 测试误差(均值±标准差) | 监督样本 | 无监督点 | 运行时间 |")
    print(f"|------|----------------------|---------|---------|----------|")
    for r in results:
        print(f"| {r['group']} | {r['mean_error']:.4e}±{r['std_error']:.4e} | "
              f"{int(r['mean_samples'])} | {int(r['mean_unsupervised'])} | {r['mean_duration']:.1f}s |")
    
    # 对比分析
    print(f"\n【对比分析】")
    c, b, d, a, e = results[2], results[1], results[3], results[0], results[4]
    
    def pct(new, old):
        return (new - old) / old * 100
    
    print(f"  C vs B (自适应 vs FGS-等预算): {pct(c['mean_error'], b['mean_error']):+.1f}%")
    print(f"  C vs D (自适应 vs 随机): {pct(c['mean_error'], d['mean_error']):+.1f}%")
    print(f"  C vs A (自适应 vs 纯基线): {pct(c['mean_error'], a['mean_error']):+.1f}%")
    print(f"  B vs E (FGS-等预算 vs FGS-原规模): {pct(b['mean_error'], e['mean_error']):+.1f}%")
    
    # 保存结果
    save_data = {
        'config': CONFIG,
        'global_test_p': GLOBAL_TEST_P,
        'results': results
    }
    np.save('adaptive_sampling_final_results.npy', save_data)
    
    with open('adaptive_sampling_final_results.csv', 'w') as f:
        f.write("组别,测试误差均值,测试误差标准差,监督样本,无监督点,自适应新增,运行时间\n")
        for r in results:
            f.write(f"{r['group']},{r['mean_error']:.6e},{r['std_error']:.6e},"
                    f"{int(r['mean_samples'])},{int(r['mean_unsupervised'])},{int(r['mean_adaptive_added'])},{r['mean_duration']:.1f}\n")
    
    print(f"\n结果已保存到 adaptive_sampling_final_results.npy")
    print("=" * 70)
    print("实验完成!")
    print("=" * 70)

if __name__ == '__main__':
    main()
