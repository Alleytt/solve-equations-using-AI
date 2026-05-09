"""测试模型在边缘参数区域的表现"""
import torch
import numpy as np
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

from src.models.low_rank_continuous_mapping import LowRankContinuousMapping
from src.utils.data_generator import generate_parametric_matrix, get_ground_truth

def test_edge_params():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    n = 256
    model = LowRankContinuousMapping(input_dim=1, hidden_dim=256, latent_dim=128,
                                     output_shape=(n, n), activation='sin').to(device)

    # 加载训练好的模型
    model.load_state_dict(torch.load('models/neumatc_n256.pth', map_location=device))
    model.eval()

    # 测试边缘参数区域 (与test_model函数相同)
    np.random.seed(42)  # 使用与测试相同的随机种子
    num_test = 20

    p_test = np.concatenate([
        np.random.uniform(0.0, 0.1, num_test // 4),    # 低参数区
        np.random.uniform(0.9, 1.0, num_test // 4),   # 高参数区
        np.random.uniform(0.1, 0.9, num_test // 2)    # 中间区
    ])
    np.random.shuffle(p_test)

    print(f"\n测试参数分布: {len(p_test)} 个样本")
    print(f"参数范围: {p_test.min():.4f} ~ {p_test.max():.4f}")

    errors = []
    cond_nums = []

    with torch.no_grad():
        for i, p in enumerate(p_test):
            H = generate_parametric_matrix(p, n).to(device)
            cond = torch.linalg.cond(H).item()
            cond_nums.append(cond)

            p_tensor = torch.tensor([[p]], dtype=torch.float32).to(device)
            pred_inv = model(p_tensor).squeeze()

            I = torch.eye(n, device=device)
            residual = H @ pred_inv - I
            rel_err = torch.norm(residual) / torch.norm(I)
            errors.append(rel_err.item())

            if i < 5:  # 只显示前5个
                print("4.1f")

    # 统计结果
    errors = np.array(errors)
    cond_nums = np.array(cond_nums)

    print("\n=== 统计结果 ===")
    print(f"平均相对误差: {errors.mean():.4f}")
    print(f"误差标准差: {errors.std():.4f}")
    print(f"最大条件数: {cond_nums.max():.2e}")
    print(f"最小条件数: {cond_nums.min():.2e}")
    print(f"平均条件数: {cond_nums.mean():.2e}")
    print(f"中位数条件数: {np.median(cond_nums):.2e}")

    # 分析高误差样本
    high_err_indices = np.where(errors > 0.5)[0]
    if len(high_err_indices) > 0:
        print("\n=== 高误差样本分析 ===")
        for idx in high_err_indices[:5]:  # 只显示前5个
            p = p_test[idx]
            err = errors[idx]
            cond = cond_nums[idx]
            print("4.1f")

if __name__ == '__main__':
    test_edge_params()