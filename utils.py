import torch


def random_linear_function(size, L=1.0):

	A = torch.randn((size, size))
	b = torch.randn((size,))

	def f(x):

		if type(x) != torch.Tensor:
			x = torch.tensor(x)

		return torch.norm(A @ x - b).pow_(2).div_(2) + L/6 * torch.norm(x).pow_(3)

	def grad(x):

		if type(x) != torch.Tensor:
			x = torch.tensor(x)

		return A.T @ (A @ x - b) + L/2 * torch.norm(x) * x  


	def hessian(x):
		if type(x) != torch.Tensor:
			x = torch.tensor(x)

		H = A.T @ A + L * (x @ x.T).div_(torch.norm(x))

		return H



	return f, grad, hessian
