import numpy as np

class MatrixExplainer:
    @staticmethod
    def generate_explanation(operation_type, A, X=None, B=None):
        lines = []
        lines.append(f"操作类型: {operation_type}")
        lines.append(f"矩阵 A 形状: {A.shape}")
        lines.append(f"条件数: {np.linalg.cond(A):.2e}")

        if operation_type == 'inv' and X is not None:
            lines.append(f"逆矩阵 X 形状: {X.shape}")
            residual = A @ X - np.eye(A.shape[0])
            lines.append(f"残差 ||AX - I||: {np.linalg.norm(residual):.2e}")

        return "\n".join(lines)