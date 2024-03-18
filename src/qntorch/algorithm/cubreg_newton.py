
import torch
from scipy.optimize import bisect

from .algorithm import Algorithm

from qntorch.utils import grad, hessian

class CubicRegNewton(Algorithm):

	def __init__(self, x0, f, tracker, L=1.0, **kwargs):

		""" Cubic Newton Optimizer

		params:
		-------
		x0: the initial point (tensor)
		f: a function that can be evaluated given a tensor (callable)
		L: the lipshitz constant of the gradient (float)
		"""

		super().__init__(x0, f, tracker, **kwargs)

		self.L = L

		return

	def _next_r(self, A, V, proj_g):
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

			# TODO: use pseudo-inverse instead of solve rmin est neg

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

		# TODO: r_min = max(0, - lambda_min(H))
		
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
		r_next, preconditioner = self._next_r(A, V, proj_g)

		# compute the update direction
		updt = V @ preconditioner

		# take a step
		x_next = x - updt

		return x_next, r_next, updt

	def step(self, x, g=None, H=None):
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

		H_x = hessian(self.f, x) if H is None else H
		g_x = grad(self.f, x) if g is None else g

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
	
class MiniCRN(Algorithm):

	def __init__(self, x0, f, tracker, g=None, H=None, M0=1.0, **kwargs):
		super().__init__(x0, f, tracker, **kwargs)
		self.M = M0
		self.g = g
		self.H = H
		return

	def _next_r(self, A, V, proj_g):
		# make idenity of the size of A
		I = torch.eye(V.size(0))
		# init preconditioner
		preconditioner = None

		# define objective function to find r
		def phi_k(r, keep_pred=False):
			# inverse
			c = torch.linalg.solve((A + (self.M/2)*r*I), proj_g)
			# save the computed preconditionner outside the function
			if keep_pred:
				nonlocal preconditioner
				preconditioner = c
			return (r - c.norm(p=2)).numpy()

		# find search interval
		r_min, r_max = self._find_r_interval(phi_k)

		# find r_k+1 as the the root of phi_k(r) and save the preconditionner
		r_next = bisect(phi_k, r_min, r_max, args=(True))

		return r_next, preconditioner

	def _find_r_interval(self, phi_k, start=1.0):
		r_min = 0
		r_max = start

		while phi_k(r_max) < 0:
			r_min = r_max
			r_max = r_max * 2

		return (r_min, r_max)

	def _update(self, x, A, V, proj_g):

		# compute r_next
		r_next, preconditioner = self._next_r(A, V, proj_g)

		# compute the update direction
		updt = V @ preconditioner

		# take a step
		x_next = x - updt

		return x_next, r_next, updt

	def step(self, x):

		# ---  compute useful quantities (one-time) ---
		H_x = hessian(self.f, x) if self.H is None else self.H
		g_x = grad(self.f, x) if self.g is None else self.g

		# compute eigen-value decomposition
		A, V = torch.linalg.eig(H_x)
		A = torch.diag(A.real)
		V = V.real

		# compute projected gradient for this step
		proj_g = V.T @ g_x

		# update rule
		x_next, _, _ = self._update(x, A, V, proj_g)

		return x_next