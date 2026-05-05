import torch
import numpy as np
import sys
import os

try:
    from .matrix_error_handler import MatrixErrorHandler
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.utils.matrix_error_handler import MatrixErrorHandler

_param_cache = {}

def generate_parametric_matrix(p, n=256, epsilon=1e-1):
    p = p.item() if isinstance(p, torch.Tensor) else p
    cache_key = (p, n)

    if cache_key not in _param_cache:
        A0 = torch.randn(n, 64) * 0.01  # 进一步缩小初始矩阵元素，降低整体尺度
        B0 = torch.randn(n, 64) * 0.01
        FA = torch.rand(64) * 1.0 + 0.5
        FB = torch.rand(64) * 1.0 + 0.5
        PhiA = torch.rand(64) * 2 * np.pi
        PhiB = torch.rand(64) * 2 * np.pi
        _param_cache[cache_key] = (A0, B0, FA, FB, PhiA, PhiB)

    A0, B0, FA, FB, PhiA, PhiB = _param_cache[cache_key]

    A = A0 * torch.sin(2 * np.pi * FA * p + PhiA)
    B = B0 * torch.cos(2 * np.pi * FB * p + PhiB)
    H = A @ B.T + epsilon * torch.eye(n)

    singular_count = 0
    while MatrixErrorHandler.is_singular(H) and singular_count < 10:
        epsilon *= 10
        H = A @ B.T + epsilon * torch.eye(n)
        singular_count += 1
        if epsilon > 1.0:
            A0_new = torch.randn(n, 64) * 0.1
            B0_new = torch.randn(n, 64) * 0.1
            A = A0_new * torch.sin(2 * np.pi * FA * p + PhiA)
            B = B0_new * torch.cos(2 * np.pi * FB * p + PhiB)
            epsilon = 1e-1
            H = A @ B.T + epsilon * torch.eye(n)

    return H


def get_ground_truth(p, n=256, op='inv'):
    H = generate_parametric_matrix(p, n)
    if op == 'inv':
        try:
            # 使用更稳定的方法计算逆
            return torch.linalg.inv(H)
        except:
            # 处理奇异矩阵
            return MatrixErrorHandler.handle_singular_matrix(H)
    elif op == 'svd':
        try:
            U, S, Vt = torch.linalg.svd(H)
            return U, S, Vt
        except:
            raise ValueError("无法计算SVD")