import torch
import matplotlib.pyplot as plt
import numpy as np

from qntorch.algorithm import T1QuasiNewton
from qntorch.utils import random_linear_function, condition_number
from qntorch.tracker import OptimTracker

import torch

if __name__ == "__main__":

	torch.set_default_dtype(torch.float64)

	# Global variables
	SIZE = 10
	M0 = 1.0
	steps = 10
	# Objects
	tracker = OptimTracker()
	f, _, _ = random_linear_function(SIZE, L=0)
	x0 = torch.randn(SIZE, requires_grad=False) # initial condition
	qn = T1QuasiNewton(x0, f, tracker, M0=M0, n_iter=steps, use_eps=False)

	# Testing the solver
	qn.solve()

	#condi_num = [condition_number(torch.tensor(X)).item() for X in qn.tracker.get("H_x")]
	"""
	plt.plot(condi_num)
	plt.xlabel("steps")
	plt.ylabel("condition number")
	plt.show()
	"""

	# Plot optim path : {f(x_k) | forall k}
	plt.semilogy(qn.tracker.get("g_norm"))
	plt.xlabel("steps")
	plt.ylabel("grad(x) norm")
	plt.show()

	plt.semilogy(qn.tracker.get("r_next"))
	plt.xlabel("steps")
	plt.ylabel("r_next")
	plt.show()

	plt.semilogy(qn.tracker.get("f_x"))
	plt.xlabel("steps")
	plt.ylabel("f_x")
	plt.show()