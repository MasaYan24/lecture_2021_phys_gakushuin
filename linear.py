import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

xmax = 1.0
sample_size = 50
err_sigma = 0.05
alpha = 1.03
beta = 0.0

np.random.seed(1)
x = np.sort(xmax * np.random.rand(sample_size))
err = err_sigma * np.random.randn(sample_size)
y = alpha * x + beta + err

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.plot(x, y, "o")
plt.show()
fig.savefig("data_plotted.png")

xy = np.stack([x, y], axis=1)
np.savetxt("data_table.csv", xy, delimiter=",")


regr = LinearRegression()
regr.fit(x.reshape(-1, 1), y.reshape(-1, 1))
xt = np.linspace(0.0, 1.0, num=101).reshape((-1, 1))
yt = regr.predict(xt)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.plot(x, y, "o")
ax.plot(xt, yt)
plt.show()
fig.savefig("data_plotted_fit.png")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.plot(x[28:31], y[28:31], "o")
xt = np.linspace(0.5, 0.6, num=10)
yt = 1.00 * xt
ax.plot(xt, yt)
plt.show()
fig.savefig("data_plotted_MSE.png")
