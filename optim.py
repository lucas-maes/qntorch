import torch
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import bisect
from tracker import OptimTracker

from torch.optim import Optimizer

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
		g_k = V.T @ self.grad(x)

		# define objective function to find r
		def phi_k(r):
			c = torch.linalg.solve((A + (self.L/2)*r*I), g_k)
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

	def bt_linesearch(self, y, x):

		f_y = self.f(y)
		f_x = self.f(x)

		g_x = self.grad(x)
		h_x = self.hessian(x)

		xydiff = y-x

		cond = f_x + g_x @ (xydiff) + (self.L/2) * (xydiff.T @ h_x @ xydiff) + (self.L/6) * xydiff.norm(p=2).pow_(3)

		if f_y <= cond:
			self.L = self.L/2
			return True
		else:
			self.L = self.L * 2
			return False

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
		g_k = V.T @ g_x

		# compute the update
		updt = V @ torch.linalg.solve((A + (self.L/2)*r_next*I) , g_k)

		# update rule
		x_next = x - updt

		# perform back-tracking line-search to find optimal L
		count = 0
		while self.bt_linesearch(x_next, x):
			count += 1
			updt = V @ torch.linalg.solve((A + (self.L/2)*r_next*I) , g_k)
			x_next = x - updt


		self.tracker(bt_linesearch_count=count,
					 L=self.L)

		return x_next

	def solve(self, x0, n_iter=100, use_eps=True, eps=10e-10):
		
		"""
		Find optimum of f(x) starting from x0

		params:
		-------
		x0: the starting point (tensor)
		n_iter: the maximum number of iteration (int)
		use_eps: use stopping criterion based on norm between last iterates (bool,)
		eps: precision to determine to stop based on the nomr of the last two iterates

		returns:
		--------
		x: the optimum find given the algorithm params (tensor)
		"""

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


class QuasiNewton(Optimizer):
	def __init__(self):
		raise NotImplementedError

class StochasticQuasiNewton(Optimizer):
	def __init__(self):
		raise NotImplementedError