# Quasi Newton PyTorch Package

under construction.

## Installation

Inside the root directory of the package `qntorch/`, run the following command:

```bash
pip3 install -e.
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
f, grad, hessian = random_linear_function(SIZE, L=L) # generate an optim problem
cn = CubicRegNewton(tracker, f, grad, hessian, L=L, n_iter=steps) # create a solver

# solve the problem
x0 = torch.randn(SIZE,)
x_star = cn.solve(x0)
print(x_star)
```

## TODOS:

- [X] Debug and make sure everything works properly  
- [ ] Adapt CubicNewton class into a clean torch.optim Optimizer class
- [ ] Run experiment to make sure it is still working when migrate to optim class
- [ ] Make algorithmic and numerical stability improvement
- [ ] Adapt code for stochasticity
