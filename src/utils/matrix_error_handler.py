import torch

class MatrixErrorHandler:
    """矩阵错误处理和鲁棒性模块"""
    
    @staticmethod
    def is_singular(matrix, tol=1e-10):
        """检测矩阵是否奇异"""
        if matrix.shape[0] != matrix.shape[1]:
            return True  # 非方阵
        
        try:
            det = torch.linalg.det(matrix)
            return abs(det) < tol
        except:
            return True
    
    @staticmethod
    def get_rank(matrix, tol=1e-10):
        """计算矩阵的秩"""
        try:
            _, S, _ = torch.linalg.svd(matrix)
            rank = torch.sum(S > tol).item()
            return rank
        except:
            return 0
    
    @staticmethod
    def analyze_equation(A, B, equation_type='AX=B'):
        """分析方程解的类型"""
        if equation_type == 'AX=B':
            # AX=B
            rank_A = MatrixErrorHandler.get_rank(A)
            rank_Aug = MatrixErrorHandler.get_rank(torch.cat([A, B], dim=1))
            
            if rank_A < rank_Aug:
                return "无解"
            elif rank_A == rank_Aug == A.shape[1]:
                return "唯一解"
            else:
                return "无穷多解"
        elif equation_type == 'XA=B':
            # XA=B -> A^T X^T = B^T
            return MatrixErrorHandler.analyze_equation(A.T, B.T, 'AX=B')
        elif equation_type == 'AXB=C':
            # AXB=C -> 先求解AY=C，再求解 Y=XB
            if not MatrixErrorHandler.is_singular(A) and not MatrixErrorHandler.is_singular(B):
                return "唯一解"
            else:
                return "可能有无解或无穷多解"
        else:
            return "未知"
    
    @staticmethod
    def handle_singular_matrix(matrix):
        """处理奇异矩阵，返回伪逆"""
        try:
            return torch.linalg.pinv(matrix)
        except:
            raise ValueError("无法处理奇异矩阵")