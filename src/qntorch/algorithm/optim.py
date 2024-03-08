import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import bisect

from torch.optim import Optimizer

class Algorithm:

	def __init__(self, tracker,
				n_iter=100,
				use_eps=True, 
				eps=10e-10
		):

		# optim tracker
		# -------------

		self.tracker = tracker      # metrics tracker

		# solver params
		# -------------

		self.n_iter = n_iter 		# max number of iterations
		self.use_eps = use_eps 		# use norm between two last iterates difference as stopping criterion
		self.eps = eps 				# precision of two last iterates difference norm to stop

		return

	def step(self, x):
		raise NotImplementedError

	def solve(self, x0):
		"""
		Find optimum of f(x) starting from x0 using self.step()

		params:
		-------
		x0: the initial point (tensor)

		returns:
		--------
		x: the optimum find given the algorithm params (tensor)
		"""

		x = x0				# current iterate
		x_last = None		# last iterate

		for step in range(self.n_iter):

			# current iterate become last iterate
			x_last = x

			# take a optimization step define our new point
			x = self.step(x)

			if self.use_eps:
				# compute norm between diff of two last iterates
				iter_dist = (x-x_last).norm(p=2)

				# log
				self.tracker(x_last=x_last,
							x=x,
							step=step+1,
							iter_dist=iter_dist.item())

				# if small enough stop training
				if iter_dist <= self.eps:
					break

		return x

	def __call__(self, x0):
		return self.solve(x0)

class CubicRegNewton(Algorithm):

	def __init__(self, tracker, f, grad, hessian, L=1.0, **kwargs):

		""" Cubic Newton Optimizer

		params:
		-------
		f: a function that can be evaluated given a tensor (callable)
		grad: the gradient of f (callable)
		hessian: the hessiance of f (callable)
		L: the lipshitz constant of the gradient (float)
		"""

		super().__init__(tracker, **kwargs)

		self.f = f
		self.grad = grad
		self.hessian = hessian
		self.L = L

		return

	def _next_r(self, x, A, V, proj_g):
		"""
		Find r_k+1 given x where r_k+1 = ||x_t+1 - x_t||

		params:
		------
		x: the current iterate (tensor)
		A: the diagonal matrix of eigenvalues of f hessian evaluated at x (tensor)
		V: the matrix containing eigenvectors of f hessian evaluated at x (tensor)
		proj_g: f gradient evaluated at x multiply by V (tensor)

		returns:
		-------
		r_next: the r_k+1 (tensor)
		preconditionner: the preconditioner computed with optimal r_next (tensor)
		"""

		# make idenity of the size of A
		I = torch.eye(V.size(0))

		# init preconditioner
		preconditioner = None

		# define objective function to find r
		def phi_k(r, keep_pred=False):

			# inverse
			c = torch.linalg.solve((A + (self.L/2)*r*I), proj_g)

			# save the computed preconditionner outside the function
			if keep_pred:
				nonlocal preconditioner
				preconditioner = c

			return (r - c.norm(p=2)).numpy()

		# find search interval
		r_min, r_max = self._find_r_interval(phi_k)

		# find r_k+1 as the the root of phi_k(r) and save the preconditionner
		r_next = bisect(phi_k, r_min, r_max, args=(True))

		# log interesting stuff
		self.tracker(search_bound=(r_min, r_max))

		return r_next, preconditioner

	def _find_r_interval(self, phi_k, start=1.0):
		"""
		Determine a search interval for finding roots of phi_k(.)

		params:
		------
		phi_k: the objective function for finding r (callable)
		start: the initial point to determine the interval (float)
		args: the args for phi_k (tuple)

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

	def _linesearch_check(self, y, x, g_x, h_x):

		f_y = self.f(y)
		f_x = self.f(x)
		xydiff = y-x
		cond = f_x + g_x @ (xydiff) + (1/2) * (xydiff.t() @ h_x @ xydiff) + (self.L/6) * xydiff.norm(p=2).pow_(3)

		return f_y <= cond

	def _update(self, x, A, V, proj_g):
		"""" Compute the update of x given necessary quantities
		
		params:
		-------
		x: the current iterate (tensor)
		A: the diagonal matrix of eigenvalues of f hessian evaluated at x (tensor)
		V: the matrix containing eigenvectors of f hessian evaluated at x (tensor)
		proj_g: f gradient evaluated at x multiply by V (tensor)

		returns:
		--------
		x_next: the updated iterate from x (tensor)
		r_next: the computed next r (float)
		"""

		# compute r_next
		r_next, preconditioner = self._next_r(x, A, V, proj_g)

		# compute the update direction
		updt = V @ preconditioner

		# take a step
		x_next = x - updt

		return x_next, r_next, updt

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

		# devise L by 2 by default
		self.L = self.L / 2

		# ---  compute useful quantities (one-time) ---

		H_x = self.hessian(x)
		g_x = self.grad(x)
		f_x = self.f(x)
		I = torch.eye(H_x.size(0))

		# compute eigen-value decomposition
		A, V = torch.linalg.eig(H_x)
		A = torch.diag(A.real)
		V = V.real

		# compute projected gradient for this step
		proj_g = V.T @ g_x

		# ---    ----    --- 

		# update rule
		x_next, r_next, update = self._update(x, A, V, proj_g)


		# perform back-tracking line-search to find optimal L
		
		count = 0
		# while condition is not satisfy increase L
		while not self._linesearch_check(x_next, x, g_x, H_x):

			count += 1

			# relax the L
			self.L = self.L * 2

			# recompute x_next with new L
			x_next, r_next, update = self._update(x, A, V, proj_g)

		# logging quantities on tracker
		self.tracker(bt_linesearch_count=count,
					 L=self.L,
					 A=A,
					 V=V,
					 f_x=f_x,
					 g_k=proj_g,
					 H_x=H_x,
					 g_x=g_x,
					 true_update=(torch.linalg.solve((H_x + (self.L/2)*r_next*I) , g_x)), 
					 g_norm=g_x.norm(p=2).item(),
					 r_next = r_next,
					 update=update)

		return x_next

class QuasiNewton(Optimizer):
	def __init__(self):
		raise NotImplementedError

class StochasticQuasiNewton(Optimizer):
	def __init__(self):
		raise NotImplementedError