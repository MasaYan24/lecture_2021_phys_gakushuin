import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

xmax = 1.0
sample_size = 50
err_sigma = 0.05
alpha = 1.03
beta = 0.0

np.random.seed(1)
x = np.sort(xmax * np.random.rand(sample_size))
err = err_sigma * np.random.randn(sample_size)
y = alpha * x + beta + err

X = torch.from_numpy(x.astype(np.float32)).view(-1, 1)
y = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

model = nn.Linear(1, 1)  # num_element_input=1, num_element_output = 1
nn.init.constant_(model.weight, 3.0)
nn.init.constant_(model.bias, 3.0)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 2000
loss_list = [2.28036132858768e-02]  # initial loss with a = 1.3
a_list = [model.state_dict()["weight"].detach().numpy()[0, 0]]
b_list = [model.state_dict()["bias"].detach().numpy()[0]]

for epoch in range(1, epochs + 1):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())
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
ax.scatter(a_list, b_list, marker="o")
ax.set_xlabel("a")
ax.set_ylabel("b")
ax.grid(True)

plt.show()
