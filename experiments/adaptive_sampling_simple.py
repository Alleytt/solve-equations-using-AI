import torch
import torch.optim as optim
import numpy as np
import time
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.low_rank_continuous_mapping import LowRankContinuousMapping
from src.utils.data_generator import generate_parametric_matrix

def run_single_group(group_name, adaptive_sampling=False, sampling_method='adaptive', seed=42):
    """运行单个实验组（简化版）"""
    n = 32
    num_train = 10
    max_iter = 200
    lambda_consist = 0.1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- {group_name} ---")
    print(f"设备: {device}, 种子: {seed}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    p_train = np.random.uniform(0, 1, num_train)
    p_test = np.random.uniform(0, 1, 50)
    p_candidate = np.random.uniform(0, 1, 500)
    
    model = LowRankContinuousMapping(input_dim=1, hidden_dim=64, latent_dim=32,
                                     output_shape=(n, n), activation='sin').to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    train_data = []
    train_p_set = set()
    
    for p in p_train:
        H = generate_parametric_matrix(p, n).to(device)
        gt = torch.linalg.solve(H.double(), torch.eye(n, device=device).double()).float().unsqueeze(0)
        train_data.append((torch.tensor([[p]], dtype=torch.float32).to(device), H.unsqueeze(0), gt))
        train_p_set.add(p)
    
    selected_points = []
    
    for it in range(max_iter):
        model.train()
        total_loss = 0.0
        
        optimizer.zero_grad()
        
        for p_tensor, H_p, gt in train_data:
            pred = model(p_tensor)
            data_loss = torch.norm(pred - gt, p='fro')**2
            I = torch.eye(n, device=device).unsqueeze(0)
            residual = torch.bmm(H_p, pred) - I
            consist_loss = torch.norm(residual, p='fro')**2 / (n*n)
            loss = data_loss + lambda_consist * consist_loss
            loss.backward()
            total_loss += loss.item()
        
        optimizer.step()
        
        # 自适应采样（每50步）
        if adaptive_sampling and (it + 1) % 50 == 0 and it > 0:
            if sampling_method == 'adaptive':
                model.eval()
                residuals = []
                with torch.no_grad():
                    for p in p_candidate:
                        if p in train_p_set:
                            continue
                        p_tensor = torch.tensor([[p]], dtype=torch.float32).to(device)
                        H = generate_parametric_matrix(p, n).to(device)
                        pred = model(p_tensor)
                        I = torch.eye(n).to(device)
                        residual = torch.norm(H @ pred.squeeze() - I) / torch.norm(I)
                        residuals.append((p, residual.item()))
                
                residuals.sort(key=lambda x: x[1], reverse=True)
                new_points = [p for p, _ in residuals[:3]]
                selected_points.extend(new_points)
            else:
                available = [p for p in p_candidate if p not in train_p_set]
                new_points = np.random.choice(available, min(3, len(available)), replace=False)
                selected_points.extend(new_points.tolist())
            
            for p in new_points:
                if p not in train_p_set:
                    H_new = generate_parametric_matrix(p, n).to(device)
                    gt_new = torch.linalg.solve(H_new.double(), torch.eye(n, device=device).double()).float().unsqueeze(0)
                    train_data.append((torch.tensor([[p]], dtype=torch.float32).to(device), H_new.unsqueeze(0), gt_new))
                    train_p_set.add(p)
        
        if it % 50 == 0:
            print(f"  迭代 {it}: 损失={total_loss/len(train_data):.4f}, 样本数={len(train_data)}")
    
    # 最终测试
    model.eval()
    with torch.no_grad():
        final_err = 0.0
        for p in p_test:
            p_tensor = torch.tensor([[p]], dtype=torch.float32).to(device)
            H = generate_parametric_matrix(p, n).to(device)
            pred = model(p_tensor)
            I = torch.eye(n).to(device)
            err = torch.norm(H @ pred.squeeze() - I) / torch.norm(I)
            final_err += err.item()
        final_err /= len(p_test)
    
    print(f"  最终测试误差: {final_err:.4e}")
    
    return {
        'group': group_name,
        'final_error': final_err,
        'final_samples': len(train_data),
        'selected_points': selected_points
    }

def main():
    print("===== 自适应采样对比实验（简化版） =====")
    
    results = []
    
    # A组：纯监督基线
    results.append(run_single_group('A组（纯监督基线）', adaptive_sampling=False))
    
    # B组：原论文FGS（简化为多样本）
    torch.manual_seed(42)
    np.random.seed(42)
    n = 32
    p_train_b = np.random.uniform(0, 1, 210)  # 10训练 + 200配点
    model_b = LowRankContinuousMapping(input_dim=1, hidden_dim=64, latent_dim=32,
                                       output_shape=(n, n), activation='sin')
    opt_b = optim.Adam(model_b.parameters(), lr=1e-4)
    train_data_b = []
    for p in p_train_b:
        H = generate_parametric_matrix(p, n)
        gt = torch.linalg.solve(H.double(), torch.eye(n).double()).float().unsqueeze(0)
        train_data_b.append((torch.tensor([[p]]), H.unsqueeze(0), gt))
    
    for it in range(200):
        model_b.train()
        total_loss = 0.0
        opt_b.zero_grad()
        for p_tensor, H_p, gt in train_data_b:
            pred = model_b(p_tensor)
            data_loss = torch.norm(pred - gt, p='fro')**2
            I = torch.eye(n).unsqueeze(0)
            residual = torch.bmm(H_p, pred) - I
            consist_loss = torch.norm(residual, p='fro')**2 / (n*n)
            loss = data_loss + 0.1 * consist_loss
            loss.backward()
            total_loss += loss.item()
        opt_b.step()
        
        if it % 50 == 0:
            print(f"  B组迭代 {it}: 损失={total_loss/len(train_data_b):.4f}, 样本数={len(train_data_b)}")
    
    model_b.eval()
    p_test = np.random.uniform(0, 1, 50)
    final_err_b = 0.0
    with torch.no_grad():
        for p in p_test:
            pred = model_b(torch.tensor([[p]]))
            H = generate_parametric_matrix(p, n)
            I = torch.eye(n)
            err = torch.norm(H @ pred.squeeze() - I) / torch.norm(I)
            final_err_b += err.item()
    final_err_b /= 50
    print(f"  B组最终测试误差: {final_err_b:.4e}")
    results.append({'group': 'B组（原论文FGS）', 'final_error': final_err_b, 'final_samples': 210, 'selected_points': []})
    
    # C组：自适应采样
    results.append(run_single_group('C组（自适应采样）', adaptive_sampling=True, sampling_method='adaptive'))
    
    # D组：随机补样
    results.append(run_single_group('D组（随机补样）', adaptive_sampling=True, sampling_method='random'))
    
    # 输出结果表格
    print("\n===== 实验结果汇总 =====")
    print("| 配置 | 测试真实误差 | 最终样本数 |")
    print("|------|------------|----------|")
    for r in results:
        print(f"| {r['group']} | {r['final_error']:.4e} | {r['final_samples']} |")
    
    # 分析C组采样点分布
    c_result = results[2]
    if c_result['selected_points']:
        selected = np.array(c_result['selected_points'])
        print(f"\nC组采样点分析:")
        print(f"  新增样本数: {len(selected)}")
        print(f"  采样点范围: [{selected.min():.3f}, {selected.max():.3f}]")
        print(f"  采样点均值: {selected.mean():.3f}")

if __name__ == '__main__':
    main()
