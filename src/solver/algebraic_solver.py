import torch
import torch.nn as nn
import sys
import os

try:
    from ..utils.matrix_error_handler import MatrixErrorHandler
    from ..utils.matrix_analyzer import MatrixAnalyzer
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.utils.matrix_error_handler import MatrixErrorHandler
    from src.utils.matrix_analyzer import MatrixAnalyzer

class AlgebraicLoss(nn.Module):
    def __init__(self, op_type='inv', lambda_consist=1.0, equation_type='AX=B'):
        super().__init__()
        self.op_type = op_type
        self.equation_type = equation_type
        self.lambda_consist = lambda_consist
        self.mse = nn.MSELoss()

    def data_loss(self, pred, target):
        return self.mse(pred, target)

    def consistency_loss(self, A_p, pred, B_p=None, C_p=None):
        if self.op_type == 'inv':
            # 求逆约束: A(p) * A_inv(p) ≈ I
            I = torch.eye(A_p.shape[1], device=A_p.device).unsqueeze(0)
            product = torch.bmm(A_p, pred)
            return self.mse(product, I)
        elif self.op_type == 'svd':
            # SVD约束: A ≈ U S V^T, U^T U≈I, V^T V≈I
            U, S, Vt = pred
            recon = torch.bmm(torch.bmm(U, torch.diag_embed(S)), Vt)
            loss_recon = self.mse(recon, A_p)
            I = torch.eye(U.shape[1], device=A_p.device).unsqueeze(0)
            loss_ortho_u = self.mse(torch.bmm(U.transpose(1,2), U), I)
            loss_ortho_v = self.mse(torch.bmm(Vt.transpose(1,2), Vt), I)
            return loss_recon + loss_ortho_u + loss_ortho_v
        elif self.equation_type == 'AX=B' and B_p is not None:
            # AX=B约束: A(p) * X(p) ≈ B(p)
            product = torch.bmm(A_p, pred)
            return self.mse(product, B_p)
        elif self.equation_type == 'XA=B' and B_p is not None:
            # XA=B约束: X(p) * A(p) ≈ B(p)
            product = torch.bmm(pred, A_p)
            return self.mse(product, B_p)
        elif self.equation_type == 'AXB=C' and B_p is not None and C_p is not None:
            # AXB=C约束: A(p) * X(p) * B(p) ≈ C(p)
            product = torch.bmm(torch.bmm(A_p, pred), B_p)
            return self.mse(product, C_p)
        else:
            return 0.0

    def forward(self, pred, target, A_p=None, B_p=None, C_p=None):
        loss_data = self.data_loss(pred, target)
        loss_consist = self.consistency_loss(A_p, pred, B_p, C_p) if A_p is not None else 0.0
        return loss_data + self.lambda_consist * loss_consist

def solve_linear_system(A, B, equation_type='AX=B'):
    """使用智能选择的方法求解线性方程组 AX=B"""
    try:
        # 使用智能算法调度
        return MatrixAnalyzer.solve_with_selector(A, B, equation_type)
    except:
        # 处理奇异矩阵
        A_pinv = MatrixErrorHandler.handle_singular_matrix(A)
        return A_pinv @ B