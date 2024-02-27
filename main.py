import torch

from xitorch.optimize import rootfinder
from utils import random_linear_function


class CubicNewton:

	def __init__(self, f, grad, hessian, L=1.0):

		self.f = f
		self.grad = grad
		self.hessian = hessian
		self.L = L

		return

	def next_r(self, x):

		# Eeigen Decomposition of the Hessian evaluated at x
		A, V = torch.linalg.eig(self.hessian(x))

		A = A.real
		V = V.real

		# make idenity of the size of A
		I = torch.eye(A.size(0))

		# compute g_k
		g_k = V @ self.grad(x)

		# define objective function to find r
		def phi_k(r):
			inv = torch.inverse((A + L/2*r*I))
			return r - torch.norm(inv @ g_k )

		# find search interval
		r_min, r_max = self.find_r_interval(phi_k)

		# start at the average of the searching interval
		x0 = (r_max - r_min) / 2

		# find r_k+1 as the the root of phi_k(r)
		r_next = rootfinder(phi_k, x0)

		return r_next

	def find_r_interval(self, phi_k, start=1.0):

		if type(start) != torch.Tensor:
			start = torch.tensor(start)

		r_max = start
		r_min = start

		while phi_k(r_max) < 0:
			r_min = r_max
			r_max = r_max * 2

		return (r_min, r_max)

	def step(self, x):

		h_x = self.hessian(x)
		g_x = self.grad(x)
		r_next = self.next_r(x)

		inv = torch.inverse(h_x + (L*r_next/2))

		x_next = x - inv @ g_x

		return x_next

if __name__ == "__main__":

	# Global variables
	SIZE = 100
	L = 10.0

	# Objects
	f, grad, hessian = random_linear_function(SIZE, L=L)
	cn = CubicNewton(f, grad, hessian, L=L)

	# Test
	x0 = torch.randn(SIZE,)

	x = x0
	last_x = None

	for step in range(100):

		last_x = x
		x = cn.step(x)

		print("||x_k+1 - x_k||: ", round(torch.norm(x - last_x).item(), 6))