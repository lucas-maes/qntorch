import torch
import qntorch as qt

if __name__ == "__main__":


    # test that the grad is the same for both method
    SIZE = 3

    f, grad, hessian = qt.utils.random_linear_function(SIZE, L=1.0)
    f_prime = qt.randlinf(SIZE)

    x = torch.randn(SIZE, requires_grad=True)

    print(qt.hessian(f, x))
    print(" ")
    print(hessian(x))

    print("(grad) analytic eq autodiff ? : ", torch.allclose(grad(x), qt.grad(f, x)))
    print("(hessian) analytic eq autodiff ? : ", torch.allclose(hessian(x), qt.hessian(f, x)))