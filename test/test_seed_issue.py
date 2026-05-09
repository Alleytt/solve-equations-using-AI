"""验证随机种子问题"""
import torch
import numpy as np
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

from src.models.low_rank_continuous_mapping import LowRankContinuousMapping
from src.utils.data_generator import generate_parametric_matrix, get_ground_truth

def test_seed_issue():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    n = 256
    model = LowRankContinuousMapping(input_dim=1, hidden_dim=256, latent_dim=128,
                                     output_shape=(n, n), activation='sin').to(device)

    # 加载训练好的模型
    model.load_state_dict(torch.load('models/neumatc_n256.pth', map_location=device))
    model.eval()

    # 模拟训练过程中的随机状态变化
    print("\n=== 模拟训练后的随机状态 ===")

    # 设置与训练开始时相同的种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 模拟训练过程中调用generate_parametric_matrix的次数
    # 训练时有40个训练样本 + 一些其他调用
    for i in range(50):  # 模拟50次调用
        _ = generate_parametric_matrix(0.5, n)

    print("随机状态已模拟训练后的状态")

    # 现在测试与test_model相同的逻辑
    num_test = 10
    p_test = np.concatenate([
        np.random.uniform(0.0, 0.1, num_test // 4),
        np.random.uniform(0.9, 1.0, num_test // 4),
        np.random.uniform(0.1, 0.9, num_test // 2)
    ])
    np.random.shuffle(p_test)

    # 设置与test_model相同的种子
    torch.manual_seed(42)
    np.random.seed(42)

    errors = []
    with torch.no_grad():
        for i, p in enumerate(p_test):
            p_tensor = torch.tensor([[p]], dtype=torch.float32).to(device)
            H = generate_parametric_matrix(p, n).to(device)

            pred = model(p_tensor)

            # 使用inv任务的测试逻辑
            gt = get_ground_truth(p, n, 'inv').to(device)
            I = torch.eye(n, device=device)
            err = torch.norm(H @ pred - I) / torch.norm(I)
            errors.append(err.item())

            if i < 3:
                print("4.1f")

    avg_error = np.mean(errors)
    print(".4f")

    # 对比：如果不模拟训练过程，直接测试
    print("\n=== 直接测试（不模拟训练过程） ===")
    torch.manual_seed(42)
    np.random.seed(42)

    p_test2 = np.concatenate([
        np.random.uniform(0.0, 0.1, num_test // 4),
        np.random.uniform(0.9, 1.0, num_test // 4),
        np.random.uniform(0.1, 0.9, num_test // 2)
    ])
    np.random.shuffle(p_test2)

    torch.manual_seed(42)
    np.random.seed(42)

    errors2 = []
    with torch.no_grad():
        for i, p in enumerate(p_test2):
            p_tensor = torch.tensor([[p]], dtype=torch.float32).to(device)
            H = generate_parametric_matrix(p, n).to(device)

            pred = model(p_tensor)

            gt = get_ground_truth(p, n, 'inv').to(device)
            I = torch.eye(n, device=device)
            err = torch.norm(H @ pred - I) / torch.norm(I)
            errors2.append(err.item())

    avg_error2 = np.mean(errors2)
    print(".4f")

if __name__ == '__main__':
    test_seed_issue()