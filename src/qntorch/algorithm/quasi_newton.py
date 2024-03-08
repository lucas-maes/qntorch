
import torch
from scipy.optimize import bisect
from .algorithm import Algorithm

from qntorch.utils import grad, hessian

class QuasiNewton(Algorithm):

	def __init__(self, tracker, f, grad, hessian, L=1.0, N=25, **kwargs):

		""" Quasi Newton Optimizer

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
		self.M = L
		self.N = N

		return

	def _linesearch_check(self, x_next, x, gx_t, Hx_t, Dt, alpha_t):

		fy = self.f(x_next)
		fx_t = self.f(x)
		Dalpha_t = Dt @ alpha_t

		cond = fx_t + gx_t.t() @ Dalpha_t + (1/2) * (alpha_t.t() @ Hx_t @ alpha_t) + (self.M/6) * Dalpha_t.norm(p=2).pow_(3)

		return fy >= cond


	def _update_Gt_eps(self, xt, Yt, Zt):
		# TODO optim: store all previous grad and compute just grad for the new element that I append
		Yt_grad = torch.tensor([grad(self.f, Yt[:, i]).div(norms[i]) for i in range(Yt.size(1))])
		Zt_grad = torch.tensor([grad(self.f, Zt[:, i]).div(norms[i]) for i in range(Zt.size(1))])
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
	
        # if # col of Dt_prev, Gt_prev, Yt_prev, Zt_prev > N, remove the first column
        mats = (Dt_prev, Gt_prev, Yt_prev, Zt_prev)
		
        for mat in mats:
            if mat.size(1) > self.N:
                mat = mat[:, 1:]
		
        Dt_prev, Gt_prev, Yt_prev, Zt_prev = mats
		
        # compute gt = grad(f, xt)
		gt = grad(self.f, xt)
		
        # compiute dt = - dtild / ||dtild||
		dtild = gt - Dt_prev @ (Dt_prev.t() @ gt)
        dt = - dtild / dtild.norm(p=2)
        
        # compute orthogonal forward estimate
        xt_half = xt + h * dt

        # append new column to Dt_prev, Gt_prev, Yt_prev, Zt_prev
        Yt = torch.cat((Yt_prev, xt_half), dim=1)
		Zt = torch.cat((Zt_prev, xt), dim=1)
        Dt = torch.cat((Dt_prev, dt), dim=1)

		Gt, eps_t = self._update_Gt_eps(xt, Yt, Zt)

        return (gt, Dt, Gt, Yt, Zt, eps_t)


	def step(self, x):

		# update Yt, Zt, Dt, Gt, eps_t
		gx_t, Dt, Gt, Yt, Zt, eps_t = self.orth_forward_estimate(x, self.L, Dt, Gt, Yt, Zt)

		# devise smoothness constant M by 2 by default
		self.M = self.M / 2

		# line search to find optimal coefficient (steps size)

		# approx hessian (make sur its symmetric)
		Hx_t = ((Gt.t() @ D) + (D.t() @ Gt)) / 2
		Hx_t = Ht_x + (self.M/2) * torch.eye(self.N) * Dt.norm()* eps_t.norm()

		# find alpha_t optimal
		alpha_t = ...

		# compute next iterate
		x_next = x + Dt @ alpha_t
		
		# perform line search
		count = 0
		# while condition is not satisfy increase L
		while not self._linesearch_check(x_next, x, gx_t, Hx_t, Dt, alpha_t):

			count += 1

			# relax the M
			self.M = self.M * 2

			# approx hessian (make sur its symmetric)
			Hx_t = ((Gt.t() @ D) + (D.t() @ Gt)) / 2
			Hx_t = Ht_x + (self.M/2) * torch.eye(self.N) * Dt.norm()* eps_t.norm()

			# find alpha_t optimal
			alpha_t = ...

			x_next = x + Dt @ alpha_t
			
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