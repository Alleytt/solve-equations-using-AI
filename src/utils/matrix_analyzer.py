import torch
import numpy as np

class MatrixAnalyzer:
    @staticmethod
    def solve_with_selector(A, B, equation_type='AX=B'):
        try:
            if equation_type == 'AX=B':
                return torch.linalg.solve(A, B)
            elif equation_type == 'XA=B':
                return torch.linalg.solve(A.T, B.T).T
            elif equation_type == 'AXB=C':
                return torch.linalg.solve(A, torch.linalg.solve(B, C))
            else:
                return torch.linalg.pinv(A) @ B
        except:
            return torch.linalg.pinv(A) @ B

    @staticmethod
    def compute_condition_number(A):
        try:
            return torch.linalg.cond(A)
        except:
            return float('inf')

    @staticmethod
    def is_well_conditioned(A, threshold=1e10):
        return MatrixAnalyzer.compute_condition_number(A) < threshold