
import torch
from qntorch.algorithm import Algorithm, CubicRegNewton
from qntorch.utils import grad

class QuasiNewton(Algorithm):

	def __init__(self, x0, f, tracker, L=1.0, N=25, **kwargs):

		""" Quasi Newton Optimizer

		params:
		-------
		x0: the initial point (tensor)
		f: a function that can be evaluated given a tensor (callable)
		L: the lipshitz constant of the gradient (float)
		"""

		super().__init__(x0, f, tracker, **kwargs)

		self.M = L
		self.N = N

		self.Dt = torch.randn((self.size, self.N))
		self.Gt = torch.randn((self.size, self.N))
		self.Yt = torch.randn((self.size, self.N))
		self.Zt = torch.randn((self.size, self.N))
		self.alpha_t = torch.randn(self.N)

		return

	def _linesearch_check(self, x_next, x, gx_t, Hx_t, Dt, alpha_t):

		fy = self.f(x_next)
		fx_t = self.f(x)
		Dalpha_t = Dt @ alpha_t

		cond = fx_t + gx_t.t() @ Dalpha_t + (1/2) * (alpha_t.t() @ Hx_t @ alpha_t) + (self.M/6) * Dalpha_t.norm(p=2).pow(3)

		return fy >= cond

	def _update_Gt_eps(self, xt, Yt, Zt):
		# TODO optim: store all previous grad and compute just grad for the new element that I append
		Yt_grad = torch.cat([grad(self.f, Yt[:, i]).unsqueeze(1) for i in range(Yt.size(1))], dim=-1)
		Zt_grad = torch.cat([grad(self.f, Zt[:, i]).unsqueeze(1) for i in range(Zt.size(1))], dim=-1)
		norms = (Yt - Zt).norm(p=2, dim=0)
		Gt = (Yt_grad - Zt_grad).div(norms.unsqueeze(0))
		eps_t = norms + 2 * (Zt - xt.unsqueeze(1)).norm(p=2, dim=0)
		return Gt, eps_t

	def orth_random_dir(self, xt, h):

		# generate random direction
		Dt = torch.randn(xt.size(0), self.N)
		Dt = torch.linalg.qr(Dt, mode='reduced').Q

		Zt = torch.repeat_interleave(xt.unsqueeze(1), self.N, dim=1)
		Yt = Zt + h * Dt

		Gt, eps_t = self._update_Gt_eps(xt, Yt, Zt)

		return (grad(self.f, xt), Dt, Gt, Yt, Zt, eps_t)

	def orth_forward_estimate(self, xt, h, Dt_prev, Gt_prev, Yt_prev, Zt_prev):

		# if # col of Dt_prev, Gt_prev, Yt_prev, Zt_prev >= N, remove the first column

		Dt_prev = Dt_prev[:, -(self.N-1):]
		Gt_prev = Gt_prev[:, -(self.N-1):]
		Yt_prev = Yt_prev[:, -(self.N-1):]
		Zt_prev = Zt_prev[:, -(self.N-1):]

		# compute gt = grad(f, xt)
		gt = grad(self.f, xt)

		# compiute dt = - dtild / ||dtild||
		dtild = gt - Dt_prev @ (Dt_prev.t() @ gt)
		dt = - dtild / dtild.norm(p=2)

		# compute orthogonal forward estimate
		xt_half = xt + h * dt


		# append new column to Dt_prev, Gt_prev, Yt_prev, Zt_prev
		Yt = torch.cat((Yt_prev, xt_half.unsqueeze(1)), dim=1)
		Zt = torch.cat((Zt_prev, xt.unsqueeze(1)), dim=1)
		Dt = torch.cat((Dt_prev, dt.unsqueeze(1)), dim=1)

		Gt, eps_t = self._update_Gt_eps(xt, Yt, Zt)

		return (gt, Dt, Gt, Yt, Zt, eps_t)

	def find_alpha(self, x, gx_t, Hx_t, Dt):
		subf = lambda alpha: self.f(x) + gx_t.t() @ (Dt @ alpha) + (1/2) * (alpha.t() @ Hx_t @ alpha) + (self.M/6) * (Dt @ alpha).norm(p=2).pow(3)
		solver = CubicRegNewton(self.alpha_t, subf, self.tracker, L=self.M) # build the solver
		return solver.solve() # solve the problem

	def step(self, x, h=10e-9):

		# update Yt, Zt, Dt, Gt, eps_t
		gx_t, self.Dt, self.Gt, self.Yt, self.Zt, eps_t = self.orth_forward_estimate(x, h, self.Dt, self.Gt, self.Yt, self.Zt)

		# devise smoothness constant M by 2 by default
		self.M = self.M / 2

		# line search to find optimal coefficient (steps size)

		# approx hessian (make sur its symmetric)
		Hx_t = ((self.Gt.t() @ self.Dt) + (self.Dt.t() @ self.Gt)) / 2
		Hx_t = Hx_t + (self.M/2) * torch.eye(self.N) * self.Dt.norm()* eps_t.norm()

		# find alpha_t optimal
		self.alpha_t = self.find_alpha(x, gx_t, Hx_t, self.Dt)

		# compute next iterate
		x_next = x + self.Dt @ self.alpha_t
		
		# perform line search
		count = 0
		# while condition is not satisfy increase L
		while not self._linesearch_check(x_next, x, gx_t, Hx_t, self.Dt, self.alpha_t):

			count += 1

			# relax the M
			self.M = self.M * 2

			# approx hessian (make sur its symmetric)
			Hx_t = ((self.Gt.t() @ self.Dt) + (self.Dt.t() @ self.Gt)) / 2
			Hx_t = Hx_t + (self.M/2) * torch.eye(self.N) * self.Dt.norm() * eps_t.norm()

			# find alpha_t optimal
			self.alpha_t = self.find_alpha(x, gx_t, Hx_t, self.Dt)

			x_next = x + self.Dt @ self.alpha_t
			
		# logging quantities on tracker
		"""
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
		"""
		return x_next