# Quasi Newton PyTorch Package

under construction.

## Installation

Inside the root directory of the package `qntorch/`, run the following command:

```bash
pip3 install -e .
```

## Usage

```python
import torch
from qntorch import CubicRegNewton, OptimTracker, random_linear_function

# set default dtype to float64
torch.set_default_dtype(torch.float64)

# problem variables
SIZE = 100
L = 1.0
steps = 100

# initialize the solver
tracker = OptimTracker()    # crate a tracker for logging
f, _, _ = random_linear_function(SIZE, L=L) # generate an optim problem
x0 = torch.randn(SIZE,) # initial condition
cn = CubicRegNewton(x0, f, tracker, L=L, n_iter=steps) # create a solver

# solve the problem
x_star = cn.solve()
print(x_star)
```

<<<<<<< HEAD
## TODO
=======
## Algorithm
- `CubRegNewton`  - Cubic Regularized Newton presented in https://link.springer.com/article/10.1007/s10107-006-0706-8
- `T1QuasiNewton` - Type-I Quasi-Newton presented in https://arxiv.org/abs/2305.19179

## TODOS:
>>>>>>> 753dad5b4648143593e16ce4e66f191c5cf9256c

### Current

- [X] [T1QN] Update the initialization of Dt, Gt, Yt, Zt, alpha_t
- [ ] [T1QN] Adapt the CubicNewton solver to find alpha_t opt
- [ ] [T1QN] Fix numerical stability issue if Hessian is not PSD (with pseudo inverse)

### Overall
- [ ] [All] Adapt algo class into a clean torch.optim Optimizer class
- [ ] [All] Run experiment to make sure it is still working when migrate to optim class vs original implem
- [ ] [All] Make algorithmic and numerical stability improvement
- [ ] [T1QN] Adapt code for stochasticity
