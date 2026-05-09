"""诊断consistloss为什么一直是0"""
import torch
import torch.optim as optim
import numpy as np
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

from src.models.low_rank_continuous_mapping import LowRankContinuousMapping
from src.utils.data_generator import generate_parametric_matrix, get_ground_truth

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 设置参数
MATRIX_SIZE = 256
n = MATRIX_SIZE
op = 'inv'
equation_type = 'inv'
num_train = 5  # 只用5个样本进行快速测试

# 生成训练数据
print("\n=== 生成训练数据 ===")
torch.manual_seed(42)
np.random.seed(42)
p_train = np.random.uniform(0, 1, num_train)

train_data = []
for p in p_train:
    H = generate_parametric_matrix(p, n).to(device)
    gt = get_ground_truth(p, n, op).to(device)
    gt = gt.unsqueeze(0).float()
    train_data.append((torch.tensor([[p]], dtype=torch.float32).to(device), H.unsqueeze(0), None, None, gt))

print(f"训练数据大小: {len(train_data)}")
print(f"H_p shape (train_data[0][1]): {train_data[0][1].shape}")
print(f"gt shape (train_data[0][4]): {train_data[0][4].shape}")

# 创建模型
model = LowRankContinuousMapping(input_dim=1, hidden_dim=256, latent_dim=128,
                                 output_shape=(n, n), activation='sin').to(device)

# Phase 1: 纯监督预训练
print("\n=== Phase 1: 纯监督预训练 ===")
phase1_optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
phase1_scheduler = optim.lr_scheduler.CosineAnnealingLR(phase1_optimizer, T_max=100)

for it in range(100):
    model.train()
    total_data_loss = 0.0
    phase1_optimizer.zero_grad()

    for p_tensor, H_p, _, _, gt in train_data:
        pred = model(p_tensor)
        data_loss = torch.norm(pred - gt, p='fro')**2
        (data_loss / len(train_data)).backward()
        total_data_loss += data_loss.item()

    phase1_optimizer.step()
    phase1_scheduler.step()

    if it % 20 == 0:
        print(f"Phase 1: iter {it}, loss: {total_data_loss/num_train:.4f}")

# Phase 2: 加入一致性约束
print("\n=== Phase 2: 一致性约束微调 ===")
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

for it in range(50):
    model.train()
    total_loss = 0.0
    total_data_loss = 0.0
    total_consist_loss = 0.0

    optimizer.zero_grad()

    for p_idx, (p_tensor, H_p, _, _, gt) in enumerate(train_data):
        pred = model(p_tensor)
        
        print(f"\n[Iter {it}, Sample {p_idx}]")
        print(f"  pred shape: {pred.shape}, dtype: {pred.dtype}")
        print(f"  H_p shape: {H_p.shape}, dtype: {H_p.dtype}")
        print(f"  gt shape: {gt.shape}, dtype: {gt.dtype}")
        print(f"  pred 值范围: [{pred.min():.4f}, {pred.max():.4f}]")
        print(f"  H_p 值范围: [{H_p.min():.4f}, {H_p.max():.4f}]")

        data_loss = torch.norm(pred - gt, p='fro')**2
        print(f"  data_loss: {data_loss.item():.6e}")

        I = torch.eye(n, device=device).unsqueeze(0)
        print(f"  I shape: {I.shape}")
        
        # 计算 residual
        residual = torch.bmm(H_p, pred) - I
        print(f"  residual shape: {residual.shape}")
        print(f"  residual 值范围: [{residual.min():.6e}, {residual.max():.6e}]")
        print(f"  residual norm: {torch.norm(residual, p='fro'):.6e}")
        
        consist_loss = torch.norm(residual, p='fro')**2 / (n*n)
        print(f"  consist_loss: {consist_loss.item():.6e}")

        loss = data_loss + 1.0 * consist_loss
        loss.backward()
        
        total_loss += loss.item()
        total_data_loss += data_loss.item()
        total_consist_loss += consist_loss.item()

    optimizer.step()

    if it % 10 == 0 or it < 5:
        avg_data_loss = total_data_loss / num_train
        avg_consist_loss = total_consist_loss / num_train
        avg_total_loss = total_loss / num_train
        print(f"\nPhase 2: iter {it}, 总损失: {avg_total_loss:.6f}, data_loss={avg_data_loss:.6e}, consist_loss={avg_consist_loss:.6e}")

print("\n=== 诊断完成 ===")
