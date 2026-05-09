"""验证训练好的模型在小矩阵上的表现"""
import torch
import numpy as np
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

from src.models.low_rank_continuous_mapping import LowRankContinuousMapping
from src.utils.data_generator import generate_parametric_matrix, get_ground_truth

def test_small_matrix():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 测试与训练相同大小的矩阵 (256x256)
    n = 256
    model = LowRankContinuousMapping(input_dim=1, hidden_dim=256, latent_dim=128,
                                     output_shape=(n, n), activation='sin').to(device)

    # 加载训练好的模型
    try:
        model.load_state_dict(torch.load('models/neumatc_n256.pth', map_location=device))
        print("成功加载模型 models/neumatc_n256.pth")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    model.eval()

    # 测试几个不同的参数
    test_params = [0.1, 0.3, 0.5, 0.7, 0.9]

    print("\n=== 测试结果 ===")
    print("参数p | 相对误差 | 条件数")
    print("-" * 30)

    with torch.no_grad():
        for p in test_params:
            # 生成测试矩阵
            H = generate_parametric_matrix(p, n).to(device)
            cond = torch.linalg.cond(H).item()

            # 获取真实逆矩阵
            gt_inv = get_ground_truth(p, n, 'inv').to(device)

            # 模型预测
            p_tensor = torch.tensor([[p]], dtype=torch.float32).to(device)
            pred_inv = model(p_tensor).squeeze()

            # 计算误差
            I = torch.eye(n, device=device)
            residual = H @ pred_inv - I
            rel_err = torch.norm(residual) / torch.norm(I)

            print("4.1f")

    # 测试训练参数范围内的表现
    print("\n=== 训练参数范围测试 (p=0.2, 0.4, 0.6, 0.8) ===")
    train_params = [0.2, 0.4, 0.6, 0.8]

    total_err = 0.0
    for p in train_params:
        H = generate_parametric_matrix(p, n).to(device)
        gt_inv = get_ground_truth(p, n, 'inv').to(device)

        p_tensor = torch.tensor([[p]], dtype=torch.float32).to(device)
        pred_inv = model(p_tensor).squeeze()

        I = torch.eye(n, device=device)
        residual = H @ pred_inv - I
        rel_err = torch.norm(residual) / torch.norm(I)
        total_err += rel_err.item()

        print("4.1f")

    avg_err = total_err / len(train_params)
    print(".4f")

if __name__ == '__main__':
    test_small_matrix()