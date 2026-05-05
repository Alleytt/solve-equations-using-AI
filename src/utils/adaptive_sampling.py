import torch
import numpy as np

def adaptive_sampling(model, A_func, candidate_p, epsilon_r=1e-3, epsilon_p=0.05, N_add=10):
    model.eval()
    with torch.no_grad():
        residuals = []
        for p in candidate_p:
            p_tensor = torch.tensor([[p]], dtype=torch.float32)
            pred = model(p_tensor)
            A_p = A_func(p).unsqueeze(0)
            # 计算残差
            I = torch.eye(A_p.shape[1]).unsqueeze(0)
            res = torch.norm(torch.bmm(A_p, pred) - I, p='fro').item()**2
            residuals.append(res)
    
    # 失败区域
    g = np.array(residuals) - epsilon_r
    fail_mask = g > 0
    fail_p = candidate_p[fail_mask]
    fail_res = g[fail_mask]
    
    # 失败概率
    fail_prob = len(fail_p) / len(candidate_p)
    
    if fail_prob > epsilon_p and len(fail_p) > 0:
        # 选残差最大的N_add个点
        idx = np.argsort(fail_res)[::-1][:N_add]
        new_col = fail_p[idx]
        return new_col, fail_prob
    else:
        return np.array([]), fail_prob