import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Callable

@torch.no_grad()
def create_contourline_figure(fn : Callable, log_scale : bool = False ) -> plt.Figure:
    x = torch.linspace(-4, 4, steps=100)
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    xy = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    z = fn(xy) if not log_scale else fn(xy).log()

    fig, ax = plt.subplots(1,1)
    ax.contour(
        xy[:,0].reshape(100, 100), 
        xy[:,1].reshape(100, 100), 
        z.reshape(100, 100),
        levels = 50
    )
    ax.scatter(1,1, label = "Minimum", c="orange", zorder=20)
    return fig

@torch.no_grad()
def scatter_path_in_figure(fig : plt.Figure, path : torch.Tensor, name : str) -> plt.Figure:
    x, y = path.split(1, dim=-1)
    ax = fig.axes[0]
    ax.set_title("Loss curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.scatter(x, y, label = name, zorder=19)
    return fig

@torch.no_grad()
def create_losscurve_figure(loss_values : list[float]) -> plt.Figure:
    fig, ax = plt.subplots(1,1)
    ax.plot(loss_values)
    return fig   

def show_figure(fig : plt.Figure) -> None:
    fig.legend()
    fig.show()