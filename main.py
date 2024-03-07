import torch
import matplotlib.pyplot as plt
import numpy as np

from optim import CubicRegNewton
from utils import random_linear_function, condition_number
from tracker import OptimTracker

import torch


# =========== PLAYGROUND =================

if __name__ == "__main__":

	torch.set_default_dtype(torch.float64)

	# Global variables
	SIZE = 10
	L = 1.0
	steps = 50

	# Objects
	tracker = OptimTracker()
	f, grad, hessian = random_linear_function(SIZE, L=L)
	cn = CubicRegNewton(tracker, f, grad, hessian, L=L, n_iter=steps)

	# Testing the solver
	x0 = torch.randn(SIZE,)
	cn.solve(x0)

	condi_num = [condition_number(torch.tensor(X)).item() for X in cn.tracker.get("H_x")]
	
	plt.plot(condi_num)
	plt.xlabel("steps")
	plt.ylabel("condition number")
	plt.show()
	
	# Plot optim path : {f(x_k) | forall k}
	plt.semilogy(cn.tracker.get("g_norm"))
	plt.xlabel("steps")
	plt.ylabel("grad(x) norm")
	plt.show()

	plt.semilogy(cn.tracker.get("r_next"))
	plt.xlabel("steps")
	plt.ylabel("r_next")
	plt.show()


	plt.semilogy(cn.tracker.get("f_x"))
	plt.xlabel("steps")
	plt.ylabel("f_x")
	plt.show()