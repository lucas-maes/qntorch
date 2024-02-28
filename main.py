import torch
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import bisect
from utils import random_linear_function
from tracker import OptimTracker

class CubicRegNewton:

	def __init__(self, tracker, f, grad, hessian, L=1.0):

		""" Cubic Newton Optimizer
		
		params:
		-------
		f: a function that can be evaluated given a tensor (callable)
		grad: the gradient of f (callable)
		hessian: the hessiance of f (callable)
		L: the lipshitz constant of the gradient (float)
		"""

		self.f = f
		self.grad = grad
		self.hessian = hessian
		self.L = L
		self.tracker = tracker

		return

	def next_r(self, x, A, V):
		"""
		Find r_k+1 given x where r_k+1 = ||x_t+1 - x_t||

		params:
		------
		x: the current iterate (tensor)

		returns:
		-------
		r_next: the r_k+1 (tensor)
		"""

		# make idenity of the size of A
		I = torch.eye(V.size(0))

		# compute g_k
		g_k = V @ self.grad(x)

		# define objective function to find r
		def phi_k(r):
			#c = torch.linalg.solve((A + (L/2)*r*I), g_k)
			c = torch.linalg.solve((self.hessian(x) + (L/2)*r*I), self.grad(x))
			return (r - c.norm(p=2)).numpy()

		# find search interval
		r_min, r_max = self.find_r_interval(phi_k)

		# find r_k+1 as the the root of phi_k(r)
		r_next = bisect(phi_k, r_min, r_max)

		# log interesting stuff
		self.tracker(search_bound=(r_min, r_max),
					 r_next=r_next,
					 A=A,
					 V=V,
					 g_k=g_k)

		return r_next

	def find_r_interval(self, phi_k, start=1.0):
		"""
		Determine a search interval for finding roots of phi_k(.)

		params:
		------
		phi_k: the objective function for finding r (callable)
		start: the initial point to determine the interval (float)

		returns:
		-------
		(r_min, r_max): a tuple containing the bound of the search interval (tensor tuple)
		"""

		r_min = 0
		r_max = start

		while phi_k(r_max) < 0:
			r_min = r_max
			r_max = r_max * 2

		return (r_min, r_max)

	def step(self, x):
		"""
		Take an cubic newton optimization step from the current iterate
	
		params:
		------
		x: the current iterate (tensor)

		returns:
		-------
		x_next: the next iterate (tensor)

		"""

		# compute useful quantities
		H_x = self.hessian(x)
		g_x = self.grad(x)
		f_x = self.f(x)
		I = torch.eye(H_x.size(0))

		# compute eigen-value decomposition
		A, V = torch.linalg.eig(H_x)
		A = torch.diag(A.real)
		V = V.real

		# compute r_k+1 and g_k
		r_next = self.next_r(x, A, V)
		g_k = V @ g_x

		# compute the update
		#updt = torch.linalg.solve((A + (L/2)*r_next*I) , g_k)
		updt = torch.linalg.solve((H_x + (L/2)*r_next*I), g_x)

		# update rule
		x_next = x - updt

		return x_next

	def solve(self, x0, n_iter=100, use_eps=True, eps=10e-10):

		x = x0
		count = 0

		for step in range(n_iter):

			count = step + 1

			last_x = x
			x = self.step(x)

			dist = round(torch.norm(x - last_x, p=2).item(), 6)

			self.tracker(last_x=last_x,
					   x=x,
					   f_x=self.f(x),
					   step=step,
					   iter_dist=dist)

			if use_eps:
				if dist <= eps:
					break

		return x


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

	cn.tracker.print("r_next")
	cn.tracker.print("search_bound")

	# Plot optim path : {f(x_k) | forall k}
	plt.semilogy(cn.tracker.get("f_x"))
	plt.show()

