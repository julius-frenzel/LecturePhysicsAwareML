import numpy as np
np.random.seed(seed=0)

n = 3 # number of physical coordinates
m = 5 # number of snapshots
r = 2 # number of reduced coordinates

S = np.random.randn(n, m)

print(f"S: {S}")

U, Sigma, V = np.linalg.svd(S)

#print(f"U: {U}")
#print(f"Sigma: {Sigma}")
#print(f"V: {V}")

Q = U[:, :r]
print(f"singular values: used: {Sigma[:r]}, unused: {Sigma[r:]}")

S_approx = Q @ Q.transpose() @ S

print(f"S_approx: {S_approx}")

e_t = np.sqrt(np.sum((S - S_approx) ** 2))
#e_t = np.linalg.norm(S - S_approx, 'fro')
print(f"true error (Frobenius norm): {e_t}")

e_c = np.sqrt(np.sum(Sigma[r:] ** 2))
print(f"computed error (sum of squares of unused singular values): {e_c}")

if n == m:
    eigvals, eigvecs = np.linalg.eig(S)
    print(f"eigenvalues for comparison: {eigvals}")




