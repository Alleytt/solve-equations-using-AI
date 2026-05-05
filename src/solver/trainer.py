import torch
import torch.optim as optim
import numpy as np
import sys
import os

try:
    from ..models.low_rank_continuous_mapping import LowRankContinuousMapping
    from ..solver.algebraic_solver import AlgebraicLoss, solve_linear_system
    from ..utils.adaptive_sampling import adaptive_sampling
    from ..utils.data_generator import generate_parametric_matrix, get_ground_truth
    from ..utils.matrix_error_handler import MatrixErrorHandler
    from ..explainer.matrix_explainer import MatrixExplainer
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.models.low_rank_continuous_mapping import LowRankContinuousMapping
    from src.solver.algebraic_solver import AlgebraicLoss, solve_linear_system
    from src.utils.adaptive_sampling import adaptive_sampling
    from src.utils.data_generator import generate_parametric_matrix, get_ground_truth
    from src.utils.matrix_error_handler import MatrixErrorHandler
    from src.explainer.matrix_explainer import MatrixExplainer

# 清除缓存以确保使用新的缩放参数
import src.utils.data_generator as data_gen_module
data_gen_module._param_cache.clear()

def train_neumatc(n=256, op='inv', equation_type='AX=B', num_train=40, num_test=100, max_iter=5000, update_T=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 设置随机种子，确保训练和测试使用相同的随机序列生成 B 矩阵
    torch.manual_seed(42)
    np.random.seed(42)

    # 采样
    p_train = np.random.uniform(0, 1, num_train)
    p_col = np.random.uniform(0, 1, 200)  # 初始配点
    p_candidate = np.random.uniform(0, 1, 1000)  # 候选采样
    
    # 模型 - 增加容量
    model = LowRankContinuousMapping(input_dim=1, hidden_dim=256, latent_dim=128,
                                     output_shape=(n, n), activation='sin').to(device)
    criterion = AlgebraicLoss(op_type=op, lambda_consist=1.0, equation_type=equation_type)

    # 降低学习率到 1e-5，使用自适应一致性损失
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
    
    # 训练数据
    train_data = []
    cond_values = []  # 记录条件数
    
    for p in p_train:
        H = generate_parametric_matrix(p, n).to(device)
        
        # 计算条件数并记录
        cond = torch.linalg.cond(H)
        cond_values.append(cond.item())
        
        # 如果条件数过大，添加正则化
        if cond > 1e5:
            H = H + 1e-5 * torch.eye(n, device=device)
            print(f"警告: 矩阵在参数p={p:.4f}时条件数过大({cond:.2e})，已添加正则化")
        
        # 检查矩阵奇异性
        if MatrixErrorHandler.is_singular(H):
            print(f"警告: 生成的矩阵在参数p={p}时是奇异的，已使用正则化处理")
        
        # 根据方程类型生成不同的训练数据
        if equation_type == 'AX=B':
            A_inv = torch.linalg.solve(H.double(), torch.eye(n, device=device).double()).float()
            I_true = torch.eye(n, device=device)
            res_gt = torch.norm(torch.mm(H, A_inv) - I_true) / n
            print(f'GT residual (inv) p={p:.2f}: {res_gt:.2e}')
            gt = A_inv.unsqueeze(0).float()
            train_data.append((torch.tensor([[p]], dtype=torch.float32).to(device), H.unsqueeze(0), None, None, gt))
        elif equation_type == 'XA=B':
            A_inv = torch.linalg.solve(H.double(), torch.eye(n, device=device).double()).float()
            I_true = torch.eye(n, device=device)
            res_gt = torch.norm(torch.mm(A_inv, H) - I_true) / n
            print(f'GT residual (inv) p={p:.2f}: {res_gt:.2e}')
            gt = A_inv.unsqueeze(0).float()
            train_data.append((torch.tensor([[p]], dtype=torch.float32).to(device), H.unsqueeze(0), None, None, gt))
        else:  # 传统的求逆
            gt = get_ground_truth(p, n, op).to(device)
            if op == 'inv':
                gt = gt.unsqueeze(0)  # 添加批次维度
                gt = gt.float()  # 转换为浮点数类型
            # Ground Truth 校验：验证 H*H^{-1} = I
            I_true = torch.eye(n, device=device)
            HX = torch.mm(H, gt[0])
            res_gt = torch.linalg.norm(HX - I_true) / n
            print(f'GT residual (inv) p={p:.2f}: {res_gt:.2e}')
            train_data.append((torch.tensor([[p]], dtype=torch.float32).to(device), H.unsqueeze(0), None, None, gt))
    
    # 条件数诊断
    if len(cond_values) > 0:
        cond_min = min(cond_values)
        cond_max = max(cond_values)
        print(f"条件数范围: {cond_min:.2e} ~ {cond_max:.2e}")
        if cond_max > 1e6:
            print("警告: 部分矩阵条件数 > 1e6，可能导致数值不稳定")

    # ========== 第一阶段：纯监督预训练 ==========
    print("\n========== Phase 1: 纯监督预训练 ==========")
    phase1_optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    phase1_scheduler = optim.lr_scheduler.CosineAnnealingLR(phase1_optimizer, T_max=1000)

    for it in range(1000):
        model.train()
        total_data_loss = 0.0

        phase1_optimizer.zero_grad()

        for p_tensor, H_p, _, _, gt in train_data:
            pred = model(p_tensor)
            data_loss = torch.norm(pred - gt, p='fro')**2
            (data_loss / len(train_data)).backward()
            total_data_loss += data_loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
        phase1_optimizer.step()
        phase1_scheduler.step()

        if it % 100 == 0:
            print(f"Phase 1: iter {it}, loss: {total_data_loss/num_train:.4f}")

    torch.save(model.state_dict(), 'phase1_inverse.pth')
    print("Phase 1 完成，模型已保存: phase1_inverse.pth")

    # ========== 第二阶段：加载权重，加入一致性约束微调 ==========
    print("\n========== Phase 2: 一致性约束微调 ==========")
    model.load_state_dict(torch.load('phase1_inverse.pth'))
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)

    avg_data_loss = None
    avg_consist_loss = None

    for it in range(max_iter):
        model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_consist_loss = 0.0

        optimizer.zero_grad()

        for p_tensor, H_p, _, _, gt in train_data:
            pred = model(p_tensor)

            data_loss = torch.norm(pred - gt, p='fro')**2

            I = torch.eye(n, device=device).unsqueeze(0)
            if equation_type == 'AX=B':
                residual = torch.bmm(H_p, pred) - I
                consist_loss = torch.norm(residual, p='fro')**2 / (n*n)
            elif equation_type == 'XA=B':
                residual = torch.bmm(pred, H_p) - I
                consist_loss = torch.norm(residual, p='fro')**2 / (n*n)
            else:
                residual = torch.bmm(H_p, pred) - I
                consist_loss = torch.norm(residual, p='fro')**2 / (n*n)

            if avg_data_loss is None:
                avg_data_loss = data_loss.item()
                avg_consist_loss = consist_loss.item()
            else:
                avg_data_loss = 0.9 * avg_data_loss + 0.1 * data_loss.item()
                avg_consist_loss = 0.9 * avg_consist_loss + 0.1 * consist_loss.item()
            lambda_consist = avg_data_loss / (avg_consist_loss + 1e-8)
            loss = data_loss + lambda_consist * consist_loss
            loss.backward()
            total_loss += loss.item()
            total_data_loss += data_loss.item()
            total_consist_loss += consist_loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
        optimizer.step()
        scheduler.step(total_loss / num_train)

        if it % update_T == 0 and it > 0:
            new_col, fail_prob = adaptive_sampling(model, lambda p: generate_parametric_matrix(p,n), p_candidate)
            if len(new_col) > 0:
                p_col = np.concatenate([p_col, new_col])

        if it % 100 == 0:
            print(f'Phase 2 迭代 {it}/{max_iter}, 总损失: {total_loss/num_train:.4f}, data_loss={total_data_loss/num_train:.4f}, consist_loss={total_consist_loss/num_train:.4f}')
            if torch.cuda.is_available():
                print(f'显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.memory_reserved()/1024**3:.2f} GB')
        else:
            if it % 1 == 0:
                print(f'Phase 2 迭代 {it}/{max_iter}, 损失: {total_loss/num_train:.4f}')

    return model

def test_model(model, n=256, num_test=100, op='inv', equation_type='AX=B', generate_explanation=False):
    device = next(model.parameters()).device
    # 使用与训练不同的参数分布：使用高条件数区域和边缘区域
    # 训练用 [0, 1]，测试用更挑战的参数：接近0和接近1的区域
    p_test = np.concatenate([
        np.random.uniform(0.0, 0.1, num_test // 4),    # 低参数区
        np.random.uniform(0.9, 1.0, num_test // 4),   # 高参数区
        np.random.uniform(0.1, 0.9, num_test // 2)    # 中间区
    ])
    np.random.shuffle(p_test)

    rel_err = 0.0
    cond_nums = []  # 记录测试时的条件数

    # 推理时间
    import time
    start = time.time()

    # 设置与训练相同的随机种子，确保测试时使用相同的 B 矩阵
    torch.manual_seed(42)
    np.random.seed(42)

    with torch.no_grad():
        for i, p in enumerate(p_test):
            p_tensor = torch.tensor([[p]], dtype=torch.float32).to(device)
            H = generate_parametric_matrix(p, n).to(device)

            # 记录并显示测试时的条件数
            cond = torch.linalg.cond(H)
            cond_nums.append(cond.item())
            if i == 0:
                print(f"测试参数范围: {p_test.min():.4f} ~ {p_test.max():.4f}")
                print(f"测试条件数范围: {min(cond_nums):.2e} ~ {max(cond_nums):.2e}")

            pred = model(p_tensor)

            if equation_type == 'AX=B':
                # pred 是学习到的逆矩阵 A^{-1}
                # 生成随机B，然后用 X = A^{-1} * B 求解
                B = torch.randn(n, n, device=device)
                X_pred = torch.mm(pred.squeeze(), B)
                err = torch.norm(torch.mm(H, X_pred) - B) / torch.norm(B)
            elif equation_type == 'XA=B':
                # pred 是学习到的逆矩阵 A^{-1}
                # 生成随机B，然后用 X = B * A^{-1} 求解
                B = torch.randn(n, n, device=device)
                X_pred = torch.mm(B, pred.squeeze())
                err = torch.norm(torch.mm(X_pred, H) - B) / torch.norm(B)
            else:  # 传统的求逆或SVD
                gt = get_ground_truth(p, n, op).to(device)
                if op == 'inv':
                    I = torch.eye(n).to(device)
                    err = torch.norm(H @ pred - I) / torch.norm(I)
                    
                    # 生成解释
                    if generate_explanation and i == 0:  # 只对第一个测试案例生成解释
                        explanation = MatrixExplainer.generate_explanation(
                            'inv', H.cpu(), X=pred.squeeze().cpu()
                        )
                        print("\n=== 求解过程解释 ===")
                        print(explanation)
                        print("====================\n")
                elif op == 'svd':
                    U, S, Vt = pred
                    recon = torch.bmm(torch.bmm(U, torch.diag_embed(S)), Vt)
                    err = torch.norm(recon - H) / torch.norm(H)
            rel_err += err.item()
    
    end = time.time()
    infer_time = (end - start) * 1000 / num_test  # ms per matrix
    rel_err /= num_test
    
    print(f'相对误差: {rel_err:.4e}')
    print(f'单矩阵推理时间: {infer_time:.2f} ms')
    return rel_err, infer_time

# 基线：NumPy直接求逆/SVD
def baseline_test(n=256, num_test=100, op='inv'):
    import time
    # 使用与训练/测试相同的随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    rel_err = 0.0
    start = time.time()
    for _ in range(num_test):
        p = np.random.uniform(0,1)
        H = generate_parametric_matrix(p, n).numpy()
        if op == 'inv':
            gt = np.linalg.inv(H)
            pred = gt
            I = np.eye(n)
            err = np.linalg.norm(H @ pred - I) / np.linalg.norm(I)
        rel_err += err
    total_time = (time.time() - start) * 1000 / num_test
    rel_err /= num_test
    print(f'基线 相对误差: {rel_err:.4e}')
    print(f'基线 单矩阵时间: {total_time:.2f} ms')
