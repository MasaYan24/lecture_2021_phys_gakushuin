import numpy as np
import matplotlib.pyplot as plt

xmax = 1.0
sample_size = 50
err_sigma = 0.1
alpha = 1.03
beta = 0.0

np.random.seed(1)
x = xmax * np.random.rand(sample_size)
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

np.savetxt("data_table.txt", x)

