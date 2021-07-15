import logging as log_module
from logging import getLogger
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from utils import retrieve_logging


def answer(alpha: float, beta: float, x: np.ndarray, err: Union[np.ndarray, float] = 0.0) -> np.ndarray:
    return alpha * x + beta + err


def get_mesh(
    x_range: Tuple[float, float], y_range: Tuple[float, float], grid: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(*x_range, grid)
    y = np.linspace(*y_range, grid)
    return np.meshgrid(x, y)


def get_loss_field(
    a_range: Tuple[float, float], b_range: Tuple[float, float], grid: int, x: np.ndarray, y_ans: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger = getLogger(__name__).getChild("get_loss_field")
    a, b = get_mesh(a_range, b_range, grid)
    diff_matrix = a[:, :, np.newaxis] * x + b[:, :, np.newaxis] - y_ans
    logger.debug(diff_matrix.shape)
    logger.debug(diff_matrix)
    logger.debug(np.mean(diff_matrix, axis=2))
    logger.debug(np.mean(diff_matrix, axis=2).shape)

    return a, b, np.mean(np.power(diff_matrix, 2), axis=2)


def create_dataset(
    xmax: float = 1.0,
    sample_size: int = 50,
    err_sigma: float = 0.05,
    alpha: float = 1.03,
    beta: float = 0.0,
    random_seed: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(random_seed)
    x = np.sort(xmax * np.random.rand(sample_size))
    err = err_sigma * np.random.randn(sample_size)
    y = answer(alpha, beta, x, err)
    return x, y


def create_model_SGD(
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    loss_list: List[float],
    a_list: List[float],
    b_list: Optional[List[float]],
    weight: float = 1.3,
    bias: Optional[float] = 1.3,
    epochs: int = 1000,
    lr: float = 0.05,
) -> nn.Linear:
    logger = getLogger(__name__).getChild("create_model_SGD")
    if bias is None:
        model = nn.Linear(1, 1, bias=False)  # num_element_input=1, num_element_output = 1
    else:
        model = nn.Linear(1, 1)  # num_element_input=1, num_element_output = 1
        nn.init.constant_(model.bias, bias)
    nn.init.constant_(model.weight, weight)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        y_pred = model(x_tensor)
        loss = criterion(y_pred, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        a_list.append(model.state_dict()["weight"].detach().numpy()[0, 0])
        if b_list is not None:
            b_list.append(model.state_dict()["bias"].detach().numpy()[0])
        if (epoch) % 100 == 0:
            logger.info(f"epoch: {epoch}, loss={loss.item():.4e}, weight={model.weight[0, 0]:.4e}")

    return model


def create_model_momentum(
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    loss_list: List[float],
    a_list: List[float],
    b_list: Optional[List[float]],
    weight: float = 1.3,
    bias: Optional[float] = 1.3,
    epochs: int = 1000,
    lr: float = 0.05,
    momentum: float = 0.9,
) -> nn.Linear:
    logger = getLogger(__name__).getChild("create_model_momentum")
    if bias is None:
        model = nn.Linear(1, 1, bias=False)  # num_element_input=1, num_element_output = 1
    else:
        model = nn.Linear(1, 1)  # num_element_input=1, num_element_output = 1
        nn.init.constant_(model.bias, bias)
    nn.init.constant_(model.weight, weight)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1):
        y_pred = model(x_tensor)
        loss = criterion(y_pred, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        a_list.append(model.state_dict()["weight"].detach().numpy()[0, 0])
        if b_list is not None:
            b_list.append(model.state_dict()["bias"].detach().numpy()[0])
        if (epoch) % 100 == 0:
            logger.info(f"epoch: {epoch}, loss={loss.item():.4e}, weight={model.weight[0, 0]:.4e}")

    return model


def create_model_Adagrad(
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    loss_list: List[float],
    a_list: List[float],
    b_list: Optional[List[float]],
    weight: float = 1.3,
    bias: Optional[float] = 1.3,
    epochs: int = 1000,
    lr: float = 0.05,
    momentum: float = 0.9,
) -> nn.Linear:
    logger = getLogger(__name__).getChild("create_model_momentum")
    if bias is None:
        model = nn.Linear(1, 1, bias=False)  # num_element_input=1, num_element_output = 1
    else:
        model = nn.Linear(1, 1)  # num_element_input=1, num_element_output = 1
        nn.init.constant_(model.bias, bias)
    nn.init.constant_(model.weight, weight)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        y_pred = model(x_tensor)
        loss = criterion(y_pred, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        a_list.append(model.state_dict()["weight"].detach().numpy()[0, 0])
        if b_list is not None:
            b_list.append(model.state_dict()["bias"].detach().numpy()[0])
        if (epoch) % 100 == 0:
            logger.info(f"epoch: {epoch}, loss={loss.item():.4e}, weight={model.weight[0, 0]:.4e}")

    return model


def main() -> None:
    logger = getLogger(__name__)
    log_module.basicConfig(level=retrieve_logging(1))
    alpha, beta = 1.03, 0.0
    x, y = create_dataset(alpha=alpha, beta=beta, random_seed=1)
    x_tensor = torch.from_numpy(x.astype(np.float32)).view(-1, 1)
    y_tensor = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

    """ linear fit only a """
    a_list = [1.3]
    loss_list = [nn.MSELoss()(y_tensor, torch.tensor(answer(a_list[0], 0.0, x)).view(-1, 1))]  # initial loss
    print("loss @a=1.3", loss_list)
    model_a = create_model_SGD(x_tensor, y_tensor, loss_list, a_list, b_list=None, weight=a_list[0], bias=None, lr=0.01)
    logger.info(f"weight={model_a.state_dict()['weight'].detach().numpy()[0, 0]}")

    # plot MSE
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(loss_list)
    ax.set_xlabel("step")
    ax.set_ylabel("MSE")
    ax.grid(True, which="both")

    axins = inset_axes(ax, width="60%", height="60%")
    axins.plot(loss_list)
    axins.set_yscale("log")
    axins.grid(True, which="both")

    plt.show()
    fig.savefig("SGD_model_a_loss.png")

    # plot a as a function of step
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(a_list)
    ax.set_xlabel("step")
    ax.set_ylabel("a")
    ax.grid(True, which="both")

    axins = inset_axes(ax, width="60%", height="60%")
    axins.plot(a_list)
    axins.set_yscale("log")
    axins.grid(True, which="both")

    plt.show()
    fig.savefig("SGD_model_a_a.png")

    # plot scatter and fitting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_tensor, y_tensor)
    xt = np.linspace(0.0, 1.3, num=10).reshape((-1, 1))
    yt = model_a(torch.tensor(xt, dtype=torch.float32))
    ax.plot(xt, yt.detach().numpy())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()
    fig.savefig("SGD_model_a_fitting.png")

    """ linear fit a and b """
    a_list = [1.3]
    b_list = [1.3]
    loss_list = [nn.MSELoss()(y_tensor, torch.tensor(answer(a_list[0], b_list[0], x)).view(-1, 1))]  # initial loss
    model_ab = create_model_SGD(x_tensor, y_tensor, loss_list, a_list, b_list, weight=a_list[0], bias=b_list[0])
    logger.info(
        f"weight={model_ab.state_dict()['weight'].detach().numpy()[0, 0]}, "
        f"bias={model_ab.state_dict()['bias'].detach().numpy()[0]}"
    )

    # plot MSE
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(loss_list)
    ax.set_xlabel("step")
    ax.set_ylabel("MSE")
    ax.grid(True, which="both")

    axins = inset_axes(ax, width="60%", height="60%")
    axins.plot(loss_list)
    axins.set_yscale("log")
    axins.grid(True, which="both")

    plt.show()
    fig.savefig("SGD_model_ab_loss.png")

    # plot a as a function of step
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(a_list)
    ax.set_xlabel("step")
    ax.set_ylabel("a")
    ax.grid(True, which="both")

    axins = inset_axes(ax, width="60%", height="40%")
    axins.plot(a_list)
    axins.set_yscale("log")
    axins.grid(True, which="both")

    plt.show()
    fig.savefig("SGD_model_ab_a.png")

    # plot b as a function of step
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(b_list)
    ax.set_xlabel("step")
    ax.set_ylabel("b")
    ax.grid(True, which="both")

    axins = inset_axes(ax, width="60%", height="60%")
    axins.plot(b_list)
    axins.set_yscale("log")
    axins.grid(True, which="both")

    plt.show()
    fig.savefig("SGD_model_ab_b.png")

    # plot a and b as a function of step
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(a_list, label="a")
    ax.plot(b_list, label="b")
    ax.set_xlabel("step")
    ax.set_ylabel("a and b")
    ax.legend()
    ax.grid(True, which="both")

    axins = inset_axes(ax, loc=7, width="60%", height="40%")
    axins.plot(a_list)
    axins.plot(b_list)
    axins.set_yscale("log")
    axins.grid(True, which="both")

    plt.show()
    fig.savefig("SGD_model_ab_ab.png")

    # plot loss_field
    a_mat, b_mat, loss_field = get_loss_field(
        a_range=(0, 3), b_range=(-1, 1.5), grid=100, x=x, y_ans=answer(alpha, beta, x)
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cont_cl = ax.contour(a_mat, b_mat, loss_field, levels=0.01 * 2 ** np.arange(10))
    ax.clabel(cont_cl, inline=True, fontsize=10)
    # ax.scatter(a_mat[0:1, 0:1], b_mat[0:1, 0:1])
    ax.scatter(1.3, 1.3)
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_xlim((0, 3))
    ax.set_ylim((-1, 1.5))
    plt.show()
    fig.savefig("SGD_model_ab_loss_field.png")

    # plot trajectory
    a_mat, b_mat, loss_field = get_loss_field(
        a_range=(0, 3), b_range=(-1, 1.5), grid=100, x=x, y_ans=answer(alpha, beta, x)
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cont_cl = ax.contour(a_mat, b_mat, loss_field, levels=0.01 * 2 ** np.arange(10))
    ax.clabel(cont_cl, inline=True, fontsize=10)
    ax.plot(a_list, b_list)
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_xlim((0, 3))
    ax.set_ylim((-1, 1.5))
    plt.show()
    fig.savefig("SGD_model_ab_trajectory.png")

    # # plot 3D surface
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # # ax.plot_surface(a_mat, b_mat, loss_field, cmap="bwr", linewidth=0)
    # cont_cl = ax.contour(a_mat, b_mat, loss_field, levels=0.01 * 2 ** np.arange(10))
    # ax.clabel(cont_cl, inline=True, fontsize=10)
    # # ax.contour(a_mat, b_mat, loss_field)
    # ax.set_xlabel("a")
    # ax.set_ylabel("b")
    # ax.set_zlabel("MSE")
    # fig.show()

    # plot scatter and fitting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_tensor, y_tensor)
    xt = np.linspace(0.0, 1.3, num=10).reshape((-1, 1))
    yt = model_ab(torch.tensor(xt, dtype=torch.float32))
    ax.plot(xt, yt.detach().numpy())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()
    fig.savefig("SGD_model_ab_fitting.png")

    """linear fit a and b with momentum"""
    a_list = [1.3]
    b_list = [1.3]
    loss_list = [nn.MSELoss()(y_tensor, torch.tensor(answer(a_list[0], b_list[0], x)).view(-1, 1))]  # initial loss
    model_ab = create_model_momentum(x_tensor, y_tensor, loss_list, a_list, b_list, weight=a_list[0], bias=b_list[0])
    logger.info(
        f"weight={model_ab.state_dict()['weight'].detach().numpy()[0, 0]}, "
        f"bias={model_ab.state_dict()['bias'].detach().numpy()[0]}"
    )

    # plot trajectory
    a_mat, b_mat, loss_field = get_loss_field(
        a_range=(0, 3), b_range=(-1, 1.5), grid=100, x=x, y_ans=answer(alpha, beta, x)
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cont_cl = ax.contour(a_mat, b_mat, loss_field, levels=0.01 * 2 ** np.arange(10))
    ax.clabel(cont_cl, inline=True, fontsize=10)
    ax.plot(a_list, b_list)
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_xlim((0, 3))
    ax.set_ylim((-1, 1.5))
    plt.show()
    # fig.savefig("momentum_model_ab_trajectory.png")

    """linear fit a and b with Adagrad"""
    a_list = [1.3]
    b_list = [1.3]
    loss_list = [nn.MSELoss()(y_tensor, torch.tensor(answer(a_list[0], b_list[0], x)).view(-1, 1))]  # initial loss
    model_ab = create_model_Adagrad(x_tensor, y_tensor, loss_list, a_list, b_list, weight=a_list[0], bias=b_list[0])
    logger.info(
        f"weight={model_ab.state_dict()['weight'].detach().numpy()[0, 0]}, "
        f"bias={model_ab.state_dict()['bias'].detach().numpy()[0]}"
    )

    # plot trajectory
    a_mat, b_mat, loss_field = get_loss_field(
        a_range=(0, 3), b_range=(-1, 1.5), grid=100, x=x, y_ans=answer(alpha, beta, x)
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cont_cl = ax.contour(a_mat, b_mat, loss_field, levels=0.01 * 2 ** np.arange(10))
    ax.clabel(cont_cl, inline=True, fontsize=10)
    ax.plot(a_list, b_list)
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_xlim((0, 3))
    ax.set_ylim((-1, 1.5))
    plt.show()
    # fig.savefig("Adagrad_model_ab_trajectory.png")


if __name__ == "__main__":
    main()
