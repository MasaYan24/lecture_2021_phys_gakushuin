from typing import Callable, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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
    a, b = get_mesh(a_range, b_range, grid)
    diff_matrix = a[:, :, np.newaxis] * x + b[:, :, np.newaxis] - y_ans
    print(diff_matrix.shape)
    print(diff_matrix)
    print(np.mean(diff_matrix, axis=2))
    print(np.mean(diff_matrix, axis=2).shape)

    return a, b, np.mean(np.power(diff_matrix, 2), axis=2)


xmax = 1.0
sample_size = 50
err_sigma = 0.05
alpha = 1.03
beta = 0.0


np.random.seed(1)
x = np.sort(xmax * np.random.rand(sample_size))
err = err_sigma * np.random.randn(sample_size)
y = answer(alpha, beta, x, err)

a_mat, b_mat, loss_field = get_loss_field(a_range=(0, 3), b_range=(-1, 1), grid=100, x=x, y_ans=answer(alpha, beta, x))

fig = plt.figure()
ax = fig.add_subplot(111)
cont_cl = ax.contour(a_mat, b_mat, loss_field, levels=[0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56])
ax.clabel(cont_cl, inline=True, fontsize=10)
plt.show()

x_tensor = torch.from_numpy(x.astype(np.float32)).view(-1, 1)
y_tensor = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

model = nn.Linear(1, 1)  # num_element_input=1, num_element_output = 1
nn.init.constant_(model.weight, 3.0)
nn.init.constant_(model.bias, 3.0)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 2000
loss_list = [torch.linalg.norm(y_tensor - answer(alpha, beta, x))]  # initial loss with a = 3.0
a_list = [model.state_dict()["weight"].detach().numpy()[0, 0]]
b_list = [model.state_dict()["bias"].detach().numpy()[0]]

for epoch in range(1, epochs + 1):
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item() / len(x))
    a_list.append(model.state_dict()["weight"].detach().numpy()[0, 0])
    b_list.append(model.state_dict()["bias"].detach().numpy()[0])
    if (epoch) % 100 == 0:
        print(f"epoch: {epoch}, loss={loss.item():.4f}, weight={model.weight[0, 0]:.8f}")

print(
    f"weight={model.state_dict()['weight'].detach().numpy()[0, 0]}, "
    f"bias={model.state_dict()['bias'].detach().numpy()[0]}"
)

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

fig.savefig("loss_SGD.png")

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
fig.savefig("a_SGD.png")


fig = plt.figure()
ax = fig.add_subplot(111)
# ax.scatter(a_list, b_list, marker="o")
ax.plot(a_list, b_list)
ax.set_xlabel("a")
ax.set_ylabel("b")
ax.grid(True)

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
cont_cl = ax.contour(a_mat, b_mat, loss_field, levels=[0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56])
ax.clabel(cont_cl, inline=True, fontsize=10)
ax.plot(a_list, b_list)
ax.set_xlabel("a")
ax.set_ylabel("b")
plt.show()
