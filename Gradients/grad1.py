import torch
import torch.nn as nn
import numpy as np

from torch.func import vjp, jvp
from torch.func import functional_call,jacrev

import time

import matplotlib.pyplot as plt

def eta(start, done, total):
    elapsed = time.time() - start
    rate = done / elapsed
    remaining = (total - done) / rate
    return elapsed, remaining


d = 3
n = 500

N = [16]
k = 2
X = np.random.normal(size=(n,d))
X_t = torch.from_numpy(X)

rho = nn.ReLU

f = nn.Sequential(
    nn.Linear(d,N[0]),
    rho(),
    nn.Linear(N[0],k)
    ).double()
theta = dict(f.named_parameters())



def f_single(theta, x):
    return functional_call(f, theta, (x.unsqueeze(0),)).squeeze(0)  # (k,)

def ntk_block(theta, x_i, x_j):
    # VJP at x_j
    _, vjp_fn = vjp(lambda th: f_single(th, x_j), theta)

    eye = torch.eye(k, dtype=x_i.dtype)

    rows = []
    for e in eye:
        # grad_theta ⟨e, f(x_j)⟩
        (grad_theta,) = vjp_fn(e)

        # JVP at x_i
        _, jvp_val = jvp(
            lambda th: f_single(th, x_i),
            (theta,),
            (grad_theta,)
        )
        rows.append(jvp_val)

    return torch.stack(rows)   # (k, k)


K = torch.zeros(n*k, n*k, dtype=torch.double)


start = time.time()
total = n * (n + 1) // 2
count = 0
for i in range(n):
    for j in range(i, n):
        block = ntk_block(theta, X_t[i], X_t[j])  # (k, k)

        count += 1

        if count % 200 == 0:
            elapsed, remaining = eta(start, count, total)
            print(
                f"{count}/{total} | "
                f"elapsed {elapsed/60:.2f} min | "
                f"ETA {remaining/60:.2f} min"
            )
        # indices in flattened kernel
        ii = slice(i * k, (i + 1) * k)
        jj = slice(j * k, (j + 1) * k)

        K[ii, jj] = block
        if i != j:
            K[jj, ii] = block.T


def Jv(theta, v_theta):
    """Compute (J v) flattened to shape (n*k,)"""
    outs = []
    for i in range(n):
        _, jvp_val = jvp(
            lambda th: f_single(th, X_t[i]),
            (theta,),
            (v_theta,)
        )
        outs.append(jvp_val)
    return torch.cat(outs)   # (n*k,)

m = 10   # number of random probes (try 20, 50, 100)

K_approx = torch.zeros(n * k, n * k, dtype=torch.double)

start = time.time()
for r in range(m):
    # random direction in parameter space
    v_theta = {
        name: torch.randn_like(p)
        for name, p in theta.items()
    }

    jv = Jv(theta, v_theta)          # (n*k,)
    K_approx += torch.outer(jv, jv)

    if (r + 1) % 10 == 0:
        elapsed = time.time() - start
        print(f"probe {r+1}/{m} | elapsed {elapsed:.2f}s")

K_approx /= m



eigvals = torch.linalg.eigvalsh(K)
eigvals_approx = torch.linalg.eigvalsh(K_approx)

eigvals = eigvals[eigvals > 1e-12]

eigvals_approx = eigvals_approx[eigvals_approx > 1e-12]

plt.plot(eigvals.detach().numpy() ,label="Hard derivative")
plt.plot(eigvals_approx.detach().numpy() ,label="Hutchinson")
plt.legend()
plt.show()


# # print(K)
# print(K.shape)
# print(eigvals)
# print(eigvals_approx)

