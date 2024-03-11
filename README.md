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

## Algorithm
- `CubRegNewton`  - Cubic Regularized Newton presented in https://link.springer.com/article/10.1007/s10107-006-0706-8
- `T1QuasiNewton` - Type-I Quasi-Newton presented in https://arxiv.org/abs/2305.19179

## TODOS:

- [X] Debug and make sure everything works properly  
- [ ] Adapt CubicNewton class into a clean torch.optim Optimizer class
- [ ] Run experiment to make sure it is still working when migrate to optim class
- [ ] Make algorithmic and numerical stability improvement
- [ ] Adapt code for stochasticity
