import torch
import torch.optim as optim
import numpy as np
import sys
import os
import time

try:
    from ..models.low_rank_continuous_mapping import LowRankContinuousMapping
    from ..solver.algebraic_solver import AlgebraicLoss, solve_linear_system
    from ..utils.adaptive_sampling import failure_informed_sampling
    from ..utils.data_generator import generate_parametric_matrix, get_ground_truth
    from ..utils.matrix_error_handler import MatrixErrorHandler
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.models.low_rank_continuous_mapping import LowRankContinuousMapping
    from src.solver.algebraic_solver import AlgebraicLoss, solve_linear_system
    from src.utils.adaptive_sampling import failure_informed_sampling
    from src.utils.data_generator import generate_parametric_matrix, get_ground_truth
    from src.utils.matrix_error_handler import MatrixErrorHandler


def train_neumatc_original(n=256, op='inv', equation_type='AX=B', 
                          num_train=40, num_test=100, max_iter=5000, 
                          update_T=100, lambda_consist=1.0, latent_dim=40,
                          hidden_dim=256, learning_rate=1e-4,
                          epsilon_r=1e-3, epsilon_p=0.05, N_add=10,
                          initial_collocation_size=200, candidate_size=1000):
    """
    Original NeuMatC implementation from the paper.
    
    Args:
        n: matrix dimension
        op: operation type ('inv' for matrix inversion)
        equation_type: equation type ('AX=B' or 'XA=B')
        num_train: number of supervised training points (N_s)
        num_test: number of test points
        max_iter: maximum iterations
        update_T: adaptive sampling interval
        lambda_consist: fixed structural consistency loss weight
        latent_dim: latent dimension (paper suggests 10~40)
        hidden_dim: MLP hidden dimension
        learning_rate: Adam learning rate
        epsilon_r: residual threshold (failure criterion)
        epsilon_p: failure probability threshold
        N_add: number of points to add per sampling
        initial_collocation_size: initial collocation set size
        candidate_size: candidate sampling set size
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    torch.manual_seed(42)
    np.random.seed(42)

    p_train = np.random.uniform(0, 1, num_train)
    p_col = np.random.uniform(0, 1, initial_collocation_size)
    p_candidate = np.random.uniform(0, 1, candidate_size)

    model = LowRankContinuousMapping(
        input_dim=1, 
        hidden_dim=hidden_dim, 
        latent_dim=latent_dim,
        output_shape=(n, n), 
        activation='sin',
        use_positional_encoding=False
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_data = []
    for p in p_train:
        H = generate_parametric_matrix(p, n).to(device)
        
        cond = torch.linalg.cond(H)
        if cond > 1e5:
            H = H + 1e-5 * torch.eye(n, device=device)
            print("Warning: Matrix at p={:.4f} has high condition number ({:.2e}), regularized".format(p, cond))

        if equation_type == 'AX=B' or equation_type == 'XA=B' or op == 'inv':
            A_inv = torch.linalg.solve(H.double(), torch.eye(n, device=device).double()).float()
            gt = A_inv.unsqueeze(0).float()
        elif op == 'svd':
            U, S, Vt = torch.linalg.svd(H)
            gt = (U, S, Vt)
        
        p_tensor = torch.tensor([[p]], dtype=torch.float32).to(device)
        train_data.append((p_tensor, H.unsqueeze(0), gt))

    collocation_data = []
    for p in p_col:
        H = generate_parametric_matrix(p, n).to(device)
        cond = torch.linalg.cond(H)
        if cond > 1e5:
            H = H + 1e-5 * torch.eye(n, device=device)
        p_tensor = torch.tensor([[p]], dtype=torch.float32).to(device)
        collocation_data.append((p_tensor, H.unsqueeze(0)))

    print("\n========== NeuMatC Single-Stage Joint Training ==========")
    print("Supervised training points:", len(train_data))
    print("Initial collocation points:", len(collocation_data))
    print("Candidate pool size:", len(p_candidate))
    print("Lambda consist:", lambda_consist)
    print("Adaptive sampling interval:", update_T)

    start_time = time.time()
    
    for it in range(max_iter):
        model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_consist_loss = 0.0

        optimizer.zero_grad()

        for p_tensor, H_p, gt in train_data:
            pred = model(p_tensor)
            data_loss = torch.norm(pred - gt, p='fro')**2 / n**2
            data_loss.backward(retain_graph=True)
            total_data_loss += data_loss.item()

        I = torch.eye(n, device=device).unsqueeze(0)
        for p_tensor, H_p in collocation_data:
            pred = model(p_tensor)
            if equation_type == 'AX=B':
                residual = torch.bmm(H_p, pred) - I
            else:
                residual = torch.bmm(pred, H_p) - I
            consist_loss = torch.norm(residual, p='fro')**2 / n**2
            (lambda_consist * consist_loss).backward(retain_graph=True)
            total_consist_loss += consist_loss.item()

        total_loss = total_data_loss + lambda_consist * total_consist_loss

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
        optimizer.step()

        if it % update_T == 0 and it > 0:
            new_col_points, fail_prob = failure_informed_sampling(
                model,
                lambda p: generate_parametric_matrix(p, n),
                p_candidate,
                epsilon_r=epsilon_r,
                epsilon_p=epsilon_p,
                N_add=N_add,
                device=device
            )
            
            if len(new_col_points) > 0:
                for new_p in new_col_points:
                    H_new = generate_parametric_matrix(new_p, n).to(device)
                    cond_new = torch.linalg.cond(H_new)
                    if cond_new > 1e5:
                        H_new = H_new + 1e-5 * torch.eye(n, device=device)
                    new_p_tensor = torch.tensor([[new_p]], dtype=torch.float32).to(device)
                    collocation_data.append((new_p_tensor, H_new.unsqueeze(0)))
                print("  Iteration {}: failure prob={:.4f}, added {} collocation points".format(it, fail_prob, len(new_col_points)))

        if it % 100 == 0:
            avg_data_loss = total_data_loss / len(train_data)
            avg_consist_loss = total_consist_loss / max(len(collocation_data), 1)
            print("Iteration {}/{}, data_loss={:.4e}, consist_loss={:.4e}, collocation_size={}".format(
                it, max_iter, avg_data_loss, avg_consist_loss, len(collocation_data)))

            if torch.cuda.is_available():
                print("  Memory usage: {:.2f} GB / {:.2f} GB".format(
                    torch.cuda.memory_allocated()/1024**3, torch.cuda.memory_reserved()/1024**3))

    train_time = time.time() - start_time
    print("\nTraining completed, time elapsed: {:.1f} s".format(train_time))

    return model


def test_neumatc(model, n=256, num_test=100, op='inv', equation_type='AX=B', test_seed=None):
    device = next(model.parameters()).device
    
    p_test = np.concatenate([
        np.random.uniform(0.0, 0.1, num_test // 4),
        np.random.uniform(0.9, 1.0, num_test // 4),
        np.random.uniform(0.1, 0.9, num_test // 2)
    ])
    np.random.shuffle(p_test)

    if test_seed is None:
        test_seed = int(time.time()) % 10000
    torch.manual_seed(test_seed)
    np.random.seed(test_seed)

    cond_nums = []
    err_list = []
    inv_err_list = []

    start = time.time()

    print("\n===== Testing Started =====")
    print("Test parameter range: {:.4f} ~ {:.4f}".format(p_test.min(), p_test.max()))
    print("Test seed:", test_seed)

    with torch.no_grad():
        for p in p_test:
            p_tensor = torch.tensor([[p]], dtype=torch.float32).to(device)
            H = generate_parametric_matrix(p, n).to(device)
            
            cond = torch.linalg.cond(H)
            if cond > 1e5:
                H = H + 1e-5 * torch.eye(n, device=device)
            cond_nums.append(cond.item())

            pred = model(p_tensor)
            I = torch.eye(n, device=device)

            inv_err = torch.norm(H @ pred - I, p='fro') / torch.norm(I)
            inv_err_list.append(inv_err.item())

            B = torch.randn(n, n, device=device)
            if equation_type == 'AX=B':
                X_pred = pred.squeeze() @ B
                solve_err = torch.norm(H @ X_pred - B, p='fro') / torch.norm(B)
            else:
                X_pred = B @ pred.squeeze()
                solve_err = torch.norm(X_pred @ H - B, p='fro') / torch.norm(B)
            err_list.append(solve_err.item())

    end = time.time()
    infer_time = (end - start) * 1000 / num_test

    avg_inv_err = np.mean(inv_err_list)
    avg_solve_err = np.mean(err_list)

    print("\n===== Test Results =====")
    print("Inverse error (||A*A_inv - I||/||I||): {:.4e}".format(avg_inv_err))
    print("Solve error: {:.4e}".format(avg_solve_err))
    print("Condition number range: {:.2e} ~ {:.2e}".format(min(cond_nums), max(cond_nums)))
    print("Inference time per matrix: {:.2f} ms".format(infer_time))

    result = {
        'inverse_error': avg_inv_err,
        'solve_error': avg_solve_err,
        'condition_numbers': cond_nums,
        'errors': err_list,
        'inverse_errors': inv_err_list,
        'parameters': p_test,
        'inference_time_ms': infer_time,
        'test_seed': test_seed,
    }

    return result


def run_neumatc_experiment(n=128, num_train=40, num_test=100, lambda_values=[0.1, 1.0, 10.0],
                          latent_dim=40, max_iter=5000, save_dir='results_neumatc'):
    os.makedirs(save_dir, exist_ok=True)
    
    results = []
    for lambda_consist in lambda_values:
        print("\n" + "="*60)
        print("Testing lambda =", lambda_consist)
        print("="*60)
        
        model = train_neumatc_original(
            n=n,
            op='inv',
            equation_type='AX=B',
            num_train=num_train,
            num_test=num_test,
            max_iter=max_iter,
            lambda_consist=lambda_consist,
            latent_dim=latent_dim
        )
        
        result = test_neumatc(model, n=n, num_test=num_test)
        result['lambda_consist'] = lambda_consist
        result['latent_dim'] = latent_dim
        
        torch.save(model.state_dict(), os.path.join(save_dir, 'neumatc_lambda_{}_model.pth'.format(lambda_consist)))
        results.append(result)
        np.save(os.path.join(save_dir, 'neumatc_lambda_{}_results.npy'.format(lambda_consist)), result)

    print("\n" + "="*60)
    print("Experiment Summary")
    print("="*60)
    for result in results:
        print("lambda={}: inverse_error={:.4e}, solve_error={:.4e}".format(
            result['lambda_consist'], result['inverse_error'], result['solve_error']))

    return results


if __name__ == '__main__':
    run_neumatc_experiment(
        n=128,
        num_train=40,
        num_test=100,
        lambda_values=[0.1, 1.0, 10.0],
        latent_dim=40,
        max_iter=5000
    )
