import torch
import torch.optim as optim
import numpy as np
import time
import os
import json
import math
from datetime import datetime

try:
    from src.models.low_rank_continuous_mapping import LowRankContinuousMapping
    from src.solver.algebraic_solver import AlgebraicLoss
    from src.utils.adaptive_sampling import adaptive_sampling, fgs_sampling, hybrid_sampling
    from src.utils.data_generator import generate_parametric_matrix, get_ground_truth
    from src.utils.matrix_error_handler import MatrixErrorHandler
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.models.low_rank_continuous_mapping import LowRankContinuousMapping
    from src.solver.algebraic_solver import AlgebraicLoss
    from src.utils.adaptive_sampling import adaptive_sampling, fgs_sampling, hybrid_sampling
    from src.utils.data_generator import generate_parametric_matrix, get_ground_truth
    from src.utils.matrix_error_handler import MatrixErrorHandler


def regularize_matrix(H, threshold=1e5, eps=1e-5):
    cond = torch.linalg.cond(H)
    if cond > threshold:
        H = H + eps * torch.eye(H.shape[0], device=H.device)
    return H, cond


def measure_true_value_time(n, device, num_trials=10):
    """测量单个矩阵真值计算的平均时间"""
    times = []
    for _ in range(num_trials):
        p = np.random.uniform(0, 1)
        H = generate_parametric_matrix(p, n).to(device)
        start = time.time()
        _ = torch.linalg.solve(H.double(), torch.eye(n, device=device).double()).float()
        times.append(time.time() - start)
    return np.mean(times), np.std(times)


def measure_epoch_time(model, train_data, optimizer, device, n, equation_type, num_trials=5):
    """测量单个训练epoch的平均时间"""
    times = []
    model.train()
    for _ in range(num_trials):
        start = time.time()
        optimizer.zero_grad()
        for p_tensor, H_p, _, _, gt in train_data:
            pred = model(p_tensor)
            data_loss = torch.norm(pred - gt, p='fro')**2
            I = torch.eye(n, device=device).unsqueeze(0)
            if equation_type == 'AX=B':
                residual = torch.bmm(H_p, pred) - I
            else:
                residual = torch.bmm(pred, H_p) - I
            consist_loss = torch.norm(residual, p='fro')**2 / n
            loss = data_loss + 0.1 * consist_loss
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
        optimizer.step()
        times.append(time.time() - start)
    return np.mean(times), np.std(times)


def scan_candidate_points(model, A_func, candidate_p, device, n):
    """扫描候选点并计算残差（用于失败检测）"""
    model.eval()
    residuals = []
    with torch.no_grad():
        for p in candidate_p:
            p_normalized = (p - 0.5) * 2.0 * math.pi
            p_tensor = torch.tensor([[p_normalized]], dtype=torch.float32).to(device)
            pred = model(p_tensor)
            A_p = A_func(p).to(device).unsqueeze(0)
            I = torch.eye(n, device=device).unsqueeze(0)
            res = torch.norm(torch.bmm(A_p, pred) - I, p='fro').item()**2
            residuals.append(res)
    return np.array(residuals)


def create_model(n, device):
    """创建模型"""
    return LowRankContinuousMapping(
        input_dim=1, hidden_dim=256, latent_dim=2048,
        output_shape=(n, n), activation='sin'
    ).to(device)


def prepare_training_data(n, num_train, device, equation_type='AX=B'):
    """准备初始训练数据"""
    p_train = np.random.uniform(0, 1, num_train)
    train_data = []
    train_p_set = set()
    
    for p in p_train:
        H = generate_parametric_matrix(p, n).to(device)
        H, _ = regularize_matrix(H)
        
        if equation_type == 'AX=B' or equation_type == 'XA=B':
            A_inv = torch.linalg.solve(H.double(), torch.eye(n, device=device).double()).float()
            gt = A_inv.unsqueeze(0).float()
        else:
            gt = torch.linalg.solve(H.double(), torch.eye(n, device=device).double()).float().unsqueeze(0)
        
        p_normalized = (p - 0.5) * 2.0 * math.pi
        train_data.append((
            torch.tensor([[p_normalized]], dtype=torch.float32).to(device),
            H.unsqueeze(0),
            None,
            None,
            gt
        ))
        train_p_set.add(p)
    
    return train_data, p_train, train_p_set


def compute_structural_consistency(model, test_p, A_func, device, n):
    """计算结构一致性指标 ||A(p)A^{-1}(p) - I||"""
    model.eval()
    residuals = []
    with torch.no_grad():
        for p in test_p:
            p_normalized = (p - 0.5) * 2.0 * math.pi
            p_tensor = torch.tensor([[p_normalized]], dtype=torch.float32).to(device)
            pred = model(p_tensor).squeeze()
            A_p = A_func(p).to(device)
            I = torch.eye(n, device=device)
            residual = torch.norm(A_p @ pred - I, p='fro') / torch.norm(I)
            residuals.append(residual.item())
    return np.mean(residuals), np.std(residuals)


def evaluate_model(model, test_p, A_func, device, n, equation_type='AX=B'):
    """评估模型性能"""
    model.eval()
    errors = []
    with torch.no_grad():
        for p in test_p:
            p_normalized = (p - 0.5) * 2.0 * math.pi
            p_tensor = torch.tensor([[p_normalized]], dtype=torch.float32).to(device)
            pred = model(p_tensor).squeeze()
            A_p = A_func(p).to(device)
            I = torch.eye(n, device=device)
            err = torch.norm(A_p @ pred - I, p='fro') / torch.norm(I)
            errors.append(err.item())
    return np.mean(errors), np.std(errors)


def run_fgs_method(
    model, train_data, unsupervised_data, p_train_set, p_candidate,
    device, n, budget_B, equation_type, t_true_or_profile,
    phase2_iter=5000, update_T=500, N_add=10, max_train_data=200
):
    """
    运行FGS方法（基线）

    参数:
        budget_B: 总预算（秒）
        t_true_or_profile: 如果是float则是预测量，如果是list则是Ours-AC的实际真值时间列表
    """
    if isinstance(t_true_or_profile, list):
        truth_time_profile = t_true_or_profile
        t_true = np.mean(truth_time_profile) if truth_time_profile else measure_true_value_time(n, device)[0]
    else:
        t_true = t_true_or_profile
        truth_time_profile = None

    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)

    avg_data_loss = None
    avg_consist_loss = None
    fixed_lambda = 0.1
    truth_time_idx = 0

    results = {
        'timestamps': [],
        'errors': [],
        'fail_probs': [],
        'num_train_points': [],
        'struct_consistency': [],
        'total_time_used': 0.0,
        'training_iterations': 0,
        'truth_computations': 0,
        'unsupervised_additions': 0
    }
    
    start_total = time.time()
    remaining_budget = budget_B
    
    for it in range(phase2_iter):
        if remaining_budget <= 0:
            break
        
        # 训练一个epoch
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_consist_loss = 0.0
        
        optimizer.zero_grad()
        batch_losses = []
        
        for p_tensor, H_p, _, _, gt in train_data:
            pred = model(p_tensor)
            data_loss = torch.norm(pred - gt, p='fro')**2
            I = torch.eye(n, device=device).unsqueeze(0)
            if equation_type == 'AX=B':
                residual = torch.bmm(H_p, pred) - I
            else:
                residual = torch.bmm(pred, H_p) - I
            consist_loss = torch.norm(residual, p='fro')**2 / n
            batch_losses.append((data_loss, consist_loss))
            total_data_loss += data_loss.item()
            total_consist_loss += consist_loss.item()
        
        count = len(batch_losses)
        if avg_data_loss is None:
            avg_data_loss = total_data_loss / count
            avg_consist_loss = total_consist_loss / count
        else:
            avg_data_loss = 0.9 * avg_data_loss + 0.1 * (total_data_loss / count)
            avg_consist_loss = 0.9 * avg_consist_loss + 0.1 * (total_consist_loss / count)
        
        for data_loss, consist_loss in batch_losses:
            loss = data_loss + fixed_lambda * consist_loss
            loss.backward()
            total_loss += loss.item()
        
        # 无监督损失
        for p_tensor, H_p in unsupervised_data:
            pred = model(p_tensor)
            I = torch.eye(n, device=device).unsqueeze(0)
            if equation_type == 'AX=B':
                residual = torch.bmm(H_p, pred) - I
            else:
                residual = torch.bmm(pred, H_p) - I
            unsupervised_loss = torch.norm(residual, p='fro')**2 / n
            (fixed_lambda * unsupervised_loss / max(len(unsupervised_data), 1)).backward()
            total_loss += unsupervised_loss.item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
        optimizer.step()
        scheduler.step(total_data_loss / len(train_data))
        
        epoch_time = time.time() - epoch_start
        remaining_budget -= epoch_time
        results['total_time_used'] = time.time() - start_total
        results['training_iterations'] = it + 1
        
        # 自适应采样周期
        if it % update_T == 0 and it > 0 and remaining_budget > 0:
            # 扫描候选点（计入预算）
            scan_start = time.time()
            residuals = scan_candidate_points(model, lambda p: generate_parametric_matrix(p, n), p_candidate, device, n)
            scan_time = time.time() - scan_start
            remaining_budget -= scan_time
            results['total_time_used'] = time.time() - start_total
            
            if remaining_budget <= 0:
                break
            
            # FGS采样 - 添加无监督配点
            new_col, fail_prob, has_new = fgs_sampling(
                model,
                lambda p: generate_parametric_matrix(p, n),
                p_candidate,
                N_add=N_add
            )
            
            if has_new and remaining_budget > 0:
                if truth_time_profile is not None and truth_time_idx < len(truth_time_profile):
                    bonus_time = truth_time_profile[truth_time_idx]
                    truth_time_idx += 1
                else:
                    bonus_time = N_add * t_true
                bonus_start = time.time()
                bonus_iterations = 0

                while time.time() - bonus_start < bonus_time and remaining_budget > 0:
                    model.train()
                    optimizer.zero_grad()

                    for p_tensor, H_p, _, _, gt in train_data:
                        pred = model(p_tensor)
                        data_loss = torch.norm(pred - gt, p='fro')**2
                        I = torch.eye(n, device=device).unsqueeze(0)
                        if equation_type == 'AX=B':
                            residual = torch.bmm(H_p, pred) - I
                        else:
                            residual = torch.bmm(pred, H_p) - I
                        consist_loss = torch.norm(residual, p='fro')**2 / n
                        loss = data_loss + fixed_lambda * consist_loss
                        loss.backward()

                    for p_tensor, H_p in unsupervised_data:
                        pred = model(p_tensor)
                        I = torch.eye(n, device=device).unsqueeze(0)
                        if equation_type == 'AX=B':
                            residual = torch.bmm(H_p, pred) - I
                        else:
                            residual = torch.bmm(pred, H_p) - I
                        unsupervised_loss = torch.norm(residual, p='fro')**2 / n
                        (fixed_lambda * unsupervised_loss / max(len(unsupervised_data), 1)).backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
                    optimizer.step()
                    bonus_iterations += 1

                bonus_elapsed = time.time() - bonus_start
                remaining_budget -= bonus_elapsed
                results['training_iterations'] += bonus_iterations

                for new_p_val in new_col:
                    if new_p_val not in p_train_set:
                        H_new = generate_parametric_matrix(new_p_val, n).to(device)
                        H_new, _ = regularize_matrix(H_new)
                        new_p_normalized = (new_p_val - 0.5) * 2.0 * math.pi
                        unsupervised_data.append((
                            torch.tensor([[new_p_normalized]], dtype=torch.float32).to(device),
                            H_new.unsqueeze(0)
                        ))
                        p_train_set.add(new_p_val)
                        results['unsupervised_additions'] += 1
                        if len(unsupervised_data) > max_train_data:
                            unsupervised_data = unsupervised_data[1:]
            
            # 记录性能指标
            results['timestamps'].append(results['total_time_used'])
            results['fail_probs'].append(fail_prob)
            results['num_train_points'].append(len(train_data))
        
        if it % 100 == 0:
            print(f"FGS - Iter {it}, Time: {results['total_time_used']:.2f}s, Remaining: {remaining_budget:.2f}s")
    
    return model, results


def run_ours_ac_method(
    model, train_data, p_train_set, p_candidate,
    device, n, budget_B, equation_type, t_true,
    phase2_iter=5000, update_T=500, N_add=10, max_train_data=200
):
    """
    运行主动纠错方法（Ours-AC）
    
    参数:
        budget_B: 总预算（秒）
        t_true: 真值计算时间
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
    
    avg_data_loss = None
    avg_consist_loss = None
    fixed_lambda = 0.1
    
    results = {
        'timestamps': [],
        'errors': [],
        'fail_probs': [],
        'num_train_points': [],
        'struct_consistency': [],
        'total_time_used': 0.0,
        'training_iterations': 0,
        'truth_computations': 0,
        'supervised_additions': 0,
        'truth_time_profile': []
    }
    
    start_total = time.time()
    remaining_budget = budget_B
    
    for it in range(phase2_iter):
        if remaining_budget <= 0:
            break
        
        # 训练一个epoch
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_consist_loss = 0.0
        
        optimizer.zero_grad()
        batch_losses = []
        
        for p_tensor, H_p, _, _, gt in train_data:
            pred = model(p_tensor)
            data_loss = torch.norm(pred - gt, p='fro')**2
            I = torch.eye(n, device=device).unsqueeze(0)
            if equation_type == 'AX=B':
                residual = torch.bmm(H_p, pred) - I
            else:
                residual = torch.bmm(pred, H_p) - I
            consist_loss = torch.norm(residual, p='fro')**2 / n
            batch_losses.append((data_loss, consist_loss))
            total_data_loss += data_loss.item()
            total_consist_loss += consist_loss.item()
        
        count = len(batch_losses)
        if avg_data_loss is None:
            avg_data_loss = total_data_loss / count
            avg_consist_loss = total_consist_loss / count
        else:
            avg_data_loss = 0.9 * avg_data_loss + 0.1 * (total_data_loss / count)
            avg_consist_loss = 0.9 * avg_consist_loss + 0.1 * (total_consist_loss / count)
        
        for data_loss, consist_loss in batch_losses:
            loss = data_loss + fixed_lambda * consist_loss
            loss.backward()
            total_loss += loss.item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
        optimizer.step()
        scheduler.step(total_data_loss / len(train_data))

        epoch_time = time.time() - epoch_start
        remaining_budget -= epoch_time
        results['total_time_used'] = time.time() - start_total
        results['training_iterations'] = it + 1

        # 自适应采样周期
        if it % update_T == 0 and it > 0 and remaining_budget > 0:
            # 扫描候选点（计入预算）
            scan_start = time.time()
            residuals = scan_candidate_points(model, lambda p: generate_parametric_matrix(p, n), p_candidate, device, n)
            scan_time = time.time() - scan_start
            remaining_budget -= scan_time
            results['total_time_used'] = time.time() - start_total
            
            if remaining_budget <= 0:
                break
            
            # 主动纠错采样 - 计算真值并添加监督样本
            new_col, fail_prob = adaptive_sampling(
                model,
                lambda p: generate_parametric_matrix(p, n),
                p_candidate,
                N_add=N_add
            )
            
            if len(new_col) > 0 and remaining_budget > 0:
                # 计算真值（计入预算）
                truth_start = time.time()
                for new_p_val in new_col:
                    if new_p_val not in p_train_set:
                        H_new = generate_parametric_matrix(new_p_val, n).to(device)
                        H_new, _ = regularize_matrix(H_new)
                        
                        if equation_type == 'AX=B' or equation_type == 'XA=B':
                            A_inv = torch.linalg.solve(H_new.double(), torch.eye(n, device=device).double()).float()
                            gt_new = A_inv.unsqueeze(0).float()
                        else:
                            gt_new = torch.linalg.solve(H_new.double(), torch.eye(n, device=device).double()).float().unsqueeze(0)
                        
                        new_p_normalized = (new_p_val - 0.5) * 2.0 * math.pi
                        train_data.append((
                            torch.tensor([[new_p_normalized]], dtype=torch.float32).to(device),
                            H_new.unsqueeze(0),
                            None,
                            None,
                            gt_new
                        ))
                        p_train_set.add(new_p_val)
                        results['truth_computations'] += 1
                        results['supervised_additions'] += 1
                        
                        if len(train_data) > max_train_data:
                            train_data = train_data[1:]
                
                truth_time = time.time() - truth_start
                remaining_budget -= truth_time
                results['truth_time_profile'].append(truth_time)
            
            # 记录性能指标
            results['timestamps'].append(results['total_time_used'])
            results['fail_probs'].append(fail_prob)
            results['num_train_points'].append(len(train_data))
        
        if it % 100 == 0:
            print(f"Ours-AC - Iter {it}, Time: {results['total_time_used']:.2f}s, Remaining: {remaining_budget:.2f}s")
    
    return model, results


def run_hybrid_method(
    model, train_data, unsupervised_data, p_train_set, p_candidate,
    device, n, budget_B, equation_type, t_true_or_profile,
    phase2_iter=5000, update_T=500, N_add=10, max_train_data=200
):
    """
    运行混合方法（Hybrid）- 随机选择无监督点转为监督样本

    参数:
        budget_B: 总预算（秒）
        t_true_or_profile: 如果是float则是预测量，如果是list则是Ours-AC的实际真值时间列表
    """
    if isinstance(t_true_or_profile, list):
        truth_time_profile = t_true_or_profile
        t_true = np.mean(truth_time_profile) if truth_time_profile else measure_true_value_time(n, device)[0]
    else:
        t_true = t_true_or_profile
        truth_time_profile = None

    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)

    avg_data_loss = None
    avg_consist_loss = None
    fixed_lambda = 0.1
    truth_time_idx = 0

    results = {
        'timestamps': [],
        'errors': [],
        'fail_probs': [],
        'num_train_points': [],
        'struct_consistency': [],
        'total_time_used': 0.0,
        'training_iterations': 0,
        'truth_computations': 0,
        'random_supervised_additions': 0,
        'truth_time_profile': []
    }
    
    start_total = time.time()
    remaining_budget = budget_B
    
    for it in range(phase2_iter):
        if remaining_budget <= 0:
            break
        
        # 训练一个epoch
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_consist_loss = 0.0
        
        optimizer.zero_grad()
        batch_losses = []
        
        for p_tensor, H_p, _, _, gt in train_data:
            pred = model(p_tensor)
            data_loss = torch.norm(pred - gt, p='fro')**2
            I = torch.eye(n, device=device).unsqueeze(0)
            if equation_type == 'AX=B':
                residual = torch.bmm(H_p, pred) - I
            else:
                residual = torch.bmm(pred, H_p) - I
            consist_loss = torch.norm(residual, p='fro')**2 / n
            batch_losses.append((data_loss, consist_loss))
            total_data_loss += data_loss.item()
            total_consist_loss += consist_loss.item()
        
        count = len(batch_losses)
        if avg_data_loss is None:
            avg_data_loss = total_data_loss / count
            avg_consist_loss = total_consist_loss / count
        else:
            avg_data_loss = 0.9 * avg_data_loss + 0.1 * (total_data_loss / count)
            avg_consist_loss = 0.9 * avg_consist_loss + 0.1 * (total_consist_loss / count)
        
        for data_loss, consist_loss in batch_losses:
            loss = data_loss + fixed_lambda * consist_loss
            loss.backward()
            total_loss += loss.item()
        
        # 无监督损失
        for p_tensor, H_p in unsupervised_data:
            pred = model(p_tensor)
            I = torch.eye(n, device=device).unsqueeze(0)
            if equation_type == 'AX=B':
                residual = torch.bmm(H_p, pred) - I
            else:
                residual = torch.bmm(pred, H_p) - I
            unsupervised_loss = torch.norm(residual, p='fro')**2 / n
            (fixed_lambda * unsupervised_loss / max(len(unsupervised_data), 1)).backward()
            total_loss += unsupervised_loss.item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
        optimizer.step()
        scheduler.step(total_data_loss / len(train_data))

        epoch_time = time.time() - epoch_start
        remaining_budget -= epoch_time
        results['total_time_used'] = time.time() - start_total
        results['training_iterations'] = it + 1

        # 自适应采样周期
        if it % update_T == 0 and it > 0 and remaining_budget > 0:
            # 扫描候选点（计入预算）
            scan_start = time.time()
            residuals = scan_candidate_points(model, lambda p: generate_parametric_matrix(p, n), p_candidate, device, n)
            scan_time = time.time() - scan_start
            remaining_budget -= scan_time
            results['total_time_used'] = time.time() - start_total
            
            if remaining_budget <= 0:
                break
            
            # 混合方法：随机选择候选点转为监督样本
            available_p = [p for p in p_candidate if p not in p_train_set]
            if len(available_p) > 0 and remaining_budget > 0:
                # 随机选择N_add个点
                np.random.shuffle(available_p)
                selected_p = available_p[:min(N_add, len(available_p))]
                
                # 计算真值（计入预算）
                truth_start = time.time()
                for new_p_val in selected_p:
                    if new_p_val not in p_train_set:
                        H_new = generate_parametric_matrix(new_p_val, n).to(device)
                        H_new, _ = regularize_matrix(H_new)
                        
                        if equation_type == 'AX=B' or equation_type == 'XA=B':
                            A_inv = torch.linalg.solve(H_new.double(), torch.eye(n, device=device).double()).float()
                            gt_new = A_inv.unsqueeze(0).float()
                        else:
                            gt_new = torch.linalg.solve(H_new.double(), torch.eye(n, device=device).double()).float().unsqueeze(0)
                        
                        new_p_normalized = (new_p_val - 0.5) * 2.0 * math.pi
                        train_data.append((
                            torch.tensor([[new_p_normalized]], dtype=torch.float32).to(device),
                            H_new.unsqueeze(0),
                            None,
                            None,
                            gt_new
                        ))
                        p_train_set.add(new_p_val)
                        results['truth_computations'] += 1
                        results['random_supervised_additions'] += 1
                        
                        if len(train_data) > max_train_data:
                            train_data = train_data[1:]
                
                truth_time = time.time() - truth_start
                remaining_budget -= truth_time
                results['truth_time_profile'].append(truth_time)

            # 计算失败概率
            fail_mask = residuals > 1e-3
            fail_prob = np.mean(fail_mask)
            
            results['timestamps'].append(results['total_time_used'])
            results['fail_probs'].append(fail_prob)
            results['num_train_points'].append(len(train_data))
        
        if it % 100 == 0:
            print(f"Hybrid - Iter {it}, Time: {results['total_time_used']:.2f}s, Remaining: {remaining_budget:.2f}s")
    
    return model, results


def phase1_pretrain(model, train_data, device, n, equation_type, phase1_iter=500):
    """第一阶段：纯监督预训练"""
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase1_iter)
    
    for it in range(phase1_iter):
        model.train()
        total_data_loss = 0.0
        
        optimizer.zero_grad()
        for p_tensor, H_p, _, _, gt in train_data:
            pred = model(p_tensor)
            data_loss = torch.norm(pred - gt, p='fro')**2
            pred_norm = torch.norm(pred, p='fro')
            gt_norm = torch.norm(gt, p='fro')
            scale_loss = (pred_norm - gt_norm.detach()) ** 2
            loss = data_loss + 0.1 * scale_loss
            loss.backward()
            total_data_loss += data_loss.item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
        optimizer.step()
        scheduler.step()
    
    return model


def run_single_experiment(
    method_name, n, budget_B, num_train, num_test,
    seed, device, equation_type='AX=B', update_T=500, N_add=10
):
    """运行单次实验"""
    print(f"\n{'='*60}")
    print(f"运行实验: {method_name}, 种子: {seed}, 预算: {budget_B}s")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data, p_train, p_train_set = prepare_training_data(n, num_train, device, equation_type)
    p_candidate = np.random.uniform(0, 1, 1000)

    model = create_model(n, device)
    model = phase1_pretrain(model, train_data, device, n, equation_type)

    t_true_mean, _ = measure_true_value_time(n, device)
    print(f"测量真值计算时间: {t_true_mean*1000:.2f} ms")

    unsupervised_data = []
    truth_time_profile = None

    if method_name == 'FGS' or method_name == 'Hybrid':
        print(f"预收集Ours-AC的实际真值计算时间...")
        model_profile = create_model(n, device)
        model_profile = phase1_pretrain(model_profile, train_data, device, n, equation_type)
        _, ours_results = run_ours_ac_method(
            model_profile, [list(t) for t in train_data], set(p_train_set), p_candidate.copy(),
            device, n, budget_B, equation_type, t_true_mean,
            update_T=update_T, N_add=N_add
        )
        truth_time_profile = ours_results.get('truth_time_profile', [])
        print(f"收集到{len(truth_time_profile)}个采样周期的实际真值计算时间")
        del model_profile

    if method_name == 'FGS':
        model, results = run_fgs_method(
            model, train_data, unsupervised_data, p_train_set, p_candidate,
            device, n, budget_B, equation_type, truth_time_profile if truth_time_profile else t_true_mean,
            update_T=update_T, N_add=N_add
        )
    elif method_name == 'Ours-AC':
        model, results = run_ours_ac_method(
            model, train_data, p_train_set, p_candidate,
            device, n, budget_B, equation_type, t_true_mean,
            update_T=update_T, N_add=N_add
        )
    elif method_name == 'Hybrid':
        model, results = run_hybrid_method(
            model, train_data, unsupervised_data, p_train_set, p_candidate,
            device, n, budget_B, equation_type, truth_time_profile if truth_time_profile else t_true_mean,
            update_T=update_T, N_add=N_add
        )
    else:
        raise ValueError(f"未知方法: {method_name}")
    
    # 生成测试数据
    p_test = np.random.uniform(0, 1, num_test)
    
    # 评估最终精度
    final_err_mean, final_err_std = evaluate_model(model, p_test, lambda p: generate_parametric_matrix(p, n), device, n, equation_type)
    struct_consist_mean, struct_consist_std = compute_structural_consistency(model, p_test, lambda p: generate_parametric_matrix(p, n), device, n)
    
    results['final_error_mean'] = final_err_mean
    results['final_error_std'] = final_err_std
    results['struct_consistency_mean'] = struct_consist_mean
    results['struct_consistency_std'] = struct_consist_std
    results['method'] = method_name
    results['seed'] = seed
    results['budget_B'] = budget_B
    results['matrix_size'] = n
    
    print(f"\n实验完成: {method_name}")
    print(f"最终相对误差: {final_err_mean:.4e} ± {final_err_std:.4e}")
    print(f"结构一致性: {struct_consist_mean:.4e} ± {struct_consist_std:.4e}")
    print(f"总时间消耗: {results['total_time_used']:.2f}s")
    print(f"训练迭代次数: {results['training_iterations']}")
    
    return results


def run_budget_controlled_experiment(
    n_list=[256],
    budget_B_list=[300],  # 预算列表（秒）
    num_train=40,
    num_test=200,
    num_repeats=10,
    methods=['FGS', 'Ours-AC', 'Hybrid'],
    equation_type='AX=B',
    update_T=500,
    N_add=10,
    save_dir='results/budget_experiment'
):
    """
    运行完整的预算控制对比实验
    
    参数:
        n_list: 矩阵大小列表
        budget_B_list: 预算列表（秒）
        num_train: 初始监督点数量
        num_test: 测试点数量
        num_repeats: 重复次数
        methods: 对比方法列表
        equation_type: 方程类型
        update_T: 自适应采样周期
        N_add: 每次采样新增点数
        save_dir: 结果保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    all_results = []
    
    for n in n_list:
        for budget_B in budget_B_list:
            print(f"\n{'='*80}")
            print(f"矩阵大小: {n}x{n}, 预算: {budget_B}秒")
            print(f"{'='*80}")
            
            for method in methods:
                for seed in range(num_repeats):
                    results = run_single_experiment(
                        method, n, budget_B, num_train, num_test,
                        seed, device, equation_type, update_T, N_add
                    )
                    all_results.append(results)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(save_dir, f'experiment_results_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n所有实验完成！结果已保存到: {results_file}")
    
    # 生成汇总报告
    generate_summary_report(all_results, save_dir, timestamp)
    
    return all_results


def generate_summary_report(results, save_dir, timestamp):
    """生成实验汇总报告"""
    report = []
    report.append("="*80)
    report.append("预算控制对比实验汇总报告")
    report.append("="*80)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 按方法和预算分组
    methods = sorted(set(r['method'] for r in results))
    budgets = sorted(set(r['budget_B'] for r in results))
    matrix_sizes = sorted(set(r['matrix_size'] for r in results))
    
    for n in matrix_sizes:
        report.append(f"矩阵大小: {n}x{n}")
        report.append("-" * 60)
        
        for budget in budgets:
            report.append(f"预算: {budget}秒")
            report.append(" " * 4 + f"{'方法':<10} {'最终误差(均值)':<20} {'最终误差(标准差)':<20} {'结构一致性':<20} {'训练迭代':<10} {'真值计算':<10}")
            report.append(" " * 4 + "-" * 100)
            
            for method in methods:
                method_results = [r for r in results 
                                if r['method'] == method and r['budget_B'] == budget and r['matrix_size'] == n]
                
                if len(method_results) == 0:
                    continue
                
                final_err_mean = np.mean([r['final_error_mean'] for r in method_results])
                final_err_std = np.mean([r['final_error_std'] for r in method_results])
                struct_consist = np.mean([r['struct_consistency_mean'] for r in method_results])
                training_iters = np.mean([r['training_iterations'] for r in method_results])
                truth_computations = np.sum([r.get('truth_computations', 0) for r in method_results])
                
                report.append(f" " * 4 + f"{method:<10} {final_err_mean:.4e} {'':<12} {final_err_std:.4e} {'':<12} {struct_consist:.4e} {'':<12} {int(training_iters):<10} {truth_computations:<10}")
            
            report.append("")
    
    # 添加效率比分析
    report.append("")
    report.append("效率比分析（Ours-AC / FGS）")
    report.append("-" * 60)
    report.append(f"{'预算':<10} {'误差比':<15} {'迭代比':<15}")
    report.append("-" * 40)
    
    for n in matrix_sizes:
        report.append(f"矩阵大小: {n}x{n}")
        for budget in budgets:
            fgs_results = [r for r in results if r['method'] == 'FGS' and r['budget_B'] == budget and r['matrix_size'] == n]
            ours_results = [r for r in results if r['method'] == 'Ours-AC' and r['budget_B'] == budget and r['matrix_size'] == n]
            
            if len(fgs_results) > 0 and len(ours_results) > 0:
                fgs_err = np.mean([r['final_error_mean'] for r in fgs_results])
                ours_err = np.mean([r['final_error_mean'] for r in ours_results])
                fgs_iters = np.mean([r['training_iterations'] for r in fgs_results])
                ours_iters = np.mean([r['training_iterations'] for r in ours_results])
                
                error_ratio = ours_err / fgs_err
                iter_ratio = ours_iters / fgs_iters
                
                report.append(f" " * 4 + f"{budget:<10} {error_ratio:.3f} {'':<12} {iter_ratio:.3f}")
    
    report_file = os.path.join(save_dir, f'experiment_report_{timestamp}.md')
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"报告已保存到: {report_file}")


if __name__ == '__main__':
    # 示例配置
    config = {
        'n_list': [256],  # [256, 512, 1024]
        'budget_B_list': [60, 120, 180],  # 预算列表（秒）
        'num_train': 40,
        'num_test': 200,
        'num_repeats': 3,  # 完整实验建议至少10次
        'methods': ['FGS', 'Ours-AC', 'Hybrid'],
        'equation_type': 'AX=B',
        'update_T': 500,
        'N_add': 10,
        'save_dir': 'results/budget_experiment'
    }
    
    print("实验配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    run_budget_controlled_experiment(**config)