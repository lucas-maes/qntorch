import torch
import math

def randlinf(size, L=1.0):
	return 

def grad(f, x):
	return torch.autograd.functional.jacobian(f, x)

def hessian(f, x):
	return torch.autograd.functional.hessian(f, x)

def random_linear_function(size, L=1.0):

	A = torch.randn((size, size))
	b = torch.rand(size,)

	def f(x):

		if type(x) != torch.Tensor:
			x = torch.tensor(x)

		return (torch.norm(A @ x - b).pow(2).div(2)) + L/6 * torch.norm(x).pow(3)

	def grad(x):

		if type(x) != torch.Tensor:
			x = torch.tensor(x)

		return (A.T @ (A @ x - b)) + (L/2) * torch.norm(x) * x  

	def hessian(x):
		if type(x) != torch.Tensor:
			x = torch.tensor(x)
		
		H = (A.T @ A) + (((L/2) * x.pow(2).div(torch.norm(x, p=2)) + (L/2) * torch.norm(x, p=2)) * torch.eye(size))

		return H

	return f, grad, hessian


def condition_number(X):

	A, _ = torch.linalg.eig(X)
	A = A.real

	lambda_max = torch.max(A, dim=-1).values
	lambda_min = torch.min(A, dim=-1).values

	return lambda_max.abs() / lambda_min.abs()

