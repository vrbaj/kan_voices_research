"""
Toy script to check the PyKAN 0.1.2 + torch installation.
"""
from kan import KAN
from kan.utils import create_dataset
import torch


if __name__ == '__main__':
    # create a KAN: 2D inputs, 1D output, and 5 hidden neurons.
    # cubic spline (k=3), 5 grid intervals (grid=5).
    model = KAN(width=[2, 5, 1], grid=5, k=3, seed=0, auto_save=False)
    # create dataset f(x,y) = exp(sin(pi*x)+y^2)
    dataset = create_dataset(lambda x: torch.exp(
        torch.sin(torch.pi*x[:, [0]]) + x[:, [1]]**2), n_var=2)
    # train the model
    model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.)
