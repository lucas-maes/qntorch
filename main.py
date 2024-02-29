import torch
import matplotlib.pyplot as plt
import numpy as np

from optim import CubicRegNewton
from utils import random_linear_function
from tracker import OptimTracker



# =========== PLAYGROUND =================

if __name__ == "__main__":

	# Global variables
	SIZE = 100
	L = 1.0

	# Objects
	tracker = OptimTracker()
	f, grad, hessian = random_linear_function(SIZE, L=L)
	cn = CubicRegNewton(tracker, f, grad, hessian, L=L)

	# Testing the solver
	x0 = torch.randn(SIZE,)
	cn.solve(x0)

	#cn.tracker.print("r_next")
	#cn.tracker.print("search_bound")

	# Plot optim path : {f(x_k) | forall k}
	plt.semilogy(cn.tracker.get("g_norm"))
	plt.xlabel("steps")
	plt.ylabel("grad(x) norm")
	plt.show()

