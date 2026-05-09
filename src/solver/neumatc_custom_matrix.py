"""
NeuMatC: Parametric Matrix Inverse (Improved Version)
- Numerical stable matrix generation
- Better network and training setup
- Expected test error < 0.01
"""
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# ================== Stable Parametric Matrix Family ==================
class StableParametricMatrixFamily:
    """Generate A(p) with controlled condition number"""
    def __init__(self, n=32, r=10, epsilon=0.1):
        self.n = n
        self.r = r
        self.epsilon = epsilon

        # Base matrices with normalized columns (L2 norm = 1)
        A0_raw = np.random.randn(n, r)
        B0_raw = np.random.randn(n, r)
        self.A0 = A0_raw / np.linalg.norm(A0_raw, axis=0, keepdims=True)
        self.B0 = B0_raw / np.linalg.norm(B0_raw, axis=0, keepdims=True)

        # Frequency and phase
        self.FA = np.random.uniform(0.5, 1.5, r)
        self.FB = np.random.uniform(0.5, 1.5, r)
        self.PhiA = np.random.uniform(0, 2*np.pi, r)
        self.PhiB = np.random.uniform(0, 2*np.pi, r)

    def generate(self, p):
        Asin = self.A0 * np.sin(2 * np.pi * self.FA * p + self.PhiA)
        Bcos = self.B0 * np.cos(2 * np.pi * self.FB * p + self.PhiB)
        A = Asin @ Bcos.T + self.epsilon * np.eye(self.n)
        return A.astype(np.float32)

    def compute_inverse(self, p):
        A = torch.from_numpy(self.generate(p))
        I = torch.eye(self.n)
        A_inv = torch.linalg.solve(A.double(), I.double()).float()
        return A_inv

# ================== Low-Rank Continuous Mapping (Same as paper) ==================
class LowRankContinuousMapping(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, latent_dim=30, output_shape=(32, 32)):
        super().__init__()
        self.output_shape = output_shape
        self.latent_dim = latent_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # C initialized with small random values
        self.C = nn.Parameter(torch.randn(output_shape[0], output_shape[1], latent_dim) * 0.1)

    def forward(self, p):
        phi = self.mlp(p)                   # (batch, latent_dim)
        B = phi.shape[0]
        n1, n2, d = self.C.shape
        C_flat = self.C.reshape(n1 * n2, d) # (n1*n2, d)
        # out = C_flat @ phi^T  for each batch
        out = torch.mm(phi, C_flat.t())     # (batch, n1*n2)
        out = out.reshape(B, n1, n2)
        return out

# ================== Training ==================
def train_model(model, matrix_family, p_train, p_collocation,
                lambda_consist=0.1, max_iter=2000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    I = torch.eye(matrix_family.n).unsqueeze(0)

    for it in range(max_iter):
        model.train()
        optimizer.zero_grad()

        # ---- Supervised loss ----
        sup_loss_val = 0.0
        for p in p_train:
            A_inv_gt = matrix_family.compute_inverse(p).unsqueeze(0)  # (1,n,n)
            pred = model(torch.tensor([[p]], dtype=torch.float32))    # (1,n,n)
            loss = torch.nn.functional.mse_loss(pred, A_inv_gt) * (matrix_family.n * matrix_family.n)
            loss.backward()
            sup_loss_val += loss.item()

        # ---- Consistency loss ----
        cons_loss_val = 0.0
        for p in p_collocation:
            A_np = matrix_family.generate(p)
            A = torch.from_numpy(A_np).unsqueeze(0)
            pred = model(torch.tensor([[p]], dtype=torch.float32))
            residual = torch.bmm(A, pred) - I
            loss = torch.norm(residual, p='fro')**2
            (lambda_consist * loss).backward()
            cons_loss_val += loss.item()

        optimizer.step()
        scheduler.step()

        if it % 200 == 0:
            print(f"Iter {it:4d}: sup_loss={sup_loss_val:.4f}, cons_loss={cons_loss_val:.4f}")

    return model

def evaluate(model, matrix_family, p_test):
    model.eval()
    errors = []
    with torch.no_grad():
        for p in p_test:
            A = torch.from_numpy(matrix_family.generate(p))
            pred = model(torch.tensor([[p]], dtype=torch.float32)).squeeze()
            residual = A @ pred - torch.eye(matrix_family.n)
            rel_err = torch.norm(residual, p='fro') / np.sqrt(matrix_family.n)
            errors.append(rel_err.item())
    return np.mean(errors), np.std(errors)

# ================== Main ==================
if __name__ == '__main__':
    n = 32
    r = 10
    epsilon = 0.1          # increased regularization
    num_train = 10
    num_collocation = 200
    lambda_consist = 0.1
    max_iter = 2000

    print("="*60)
    print("Improved NeuMatC: Parametric Matrix Inverse")
    print("="*60)

    matrix_family = StableParametricMatrixFamily(n=n, r=r, epsilon=epsilon)

    # Data
    np.random.seed(42)
    p_train = np.random.uniform(0, 1, num_train)
    p_collocation = np.random.uniform(0, 1, num_collocation)
    p_test = np.random.uniform(0, 1, 30)

    # NeuMatC (with consistency)
    print("\n--- Training NeuMatC ---")
    model = LowRankContinuousMapping(output_shape=(n, n), hidden_dim=128, latent_dim=30)
    model = train_model(model, matrix_family, p_train, p_collocation,
                        lambda_consist=lambda_consist, max_iter=max_iter, lr=1e-3)

    mean_err, std_err = evaluate(model, matrix_family, p_test)
    print(f"\nNeuMatC Error: {mean_err:.6e} ± {std_err:.6e}")

    # Baseline (supervised only)
    print("\n--- Training Baseline (supervised only) ---")
    model_base = LowRankContinuousMapping(output_shape=(n, n), hidden_dim=128, latent_dim=30)
    model_base = train_model(model_base, matrix_family, p_train, [],
                             lambda_consist=0.0, max_iter=max_iter, lr=1e-3)

    mean_err_base, std_err_base = evaluate(model_base, matrix_family, p_test)
    print(f"\nBaseline Error: {mean_err_base:.6e} ± {std_err_base:.6e}")

    improvement = (1 - mean_err / (mean_err_base + 1e-10)) * 100
    print(f"\nImprovement: {improvement:.1f}%")