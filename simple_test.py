import torch
import numpy as np

# 极简测试
n = 256
device = 'cpu'

# 创建虚拟数据
H_p = torch.randn(1, n, n)  # shape (1, n, n)
pred = torch.randn(1, n, n)  # shape (1, n, n)
I = torch.eye(n).unsqueeze(0)  # shape (1, n, n)

print(f"H_p shape: {H_p.shape}")
print(f"pred shape: {pred.shape}")
print(f"I shape: {I.shape}")

# 计算residual
residual = torch.bmm(H_p, pred) - I
print(f"residual shape: {residual.shape}")
print(f"residual norm: {torch.norm(residual, p='fro'):.6e}")

# 计算consist_loss
consist_loss = torch.norm(residual, p='fro')**2 / (n*n)
print(f"consist_loss (raw): {consist_loss:.6e}")
print(f"consist_loss (%.4f): {consist_loss:.4f}")
print(f"consist_loss (formatted): {consist_loss.item():.6e}")

# 检查数值范围
print(f"\nresidual max: {residual.max():.6e}")
print(f"residual min: {residual.min():.6e}")
print(f"residual mean: {residual.mean():.6e}")

# 测试当pred=H^-1时
H_inv = torch.linalg.inv(H_p)
residual_perfect = torch.bmm(H_p, H_inv) - I
consist_loss_perfect = torch.norm(residual_perfect, p='fro')**2 / (n*n)
print(f"\n当pred=H^-1时:")
print(f"residual norm (perfect): {torch.norm(residual_perfect, p='fro'):.6e}")
print(f"consist_loss (perfect): {consist_loss_perfect:.6e}")
