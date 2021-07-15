from typing import List

import matplotlib.pyplot as plt
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
# import numpy as np
from torch import nn, optim

digits_data = datasets.load_digits()

n_img = 10
# plt.figure(figsize=(10, 4))
# for i in range(n_img):
#    ax = plt.subplot(2, 5, i + 1)
#    plt.imshow(digits_data.data[i].reshape(8, 8), cmap="Greys_r")
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
# plt.show()

print(f"shape = {digits_data.data.shape}")
print(f"label: {digits_data.target[:n_img]}")


digit_images = digits_data.data
labels = digits_data.target
x_train, x_test, t_train, t_test = train_test_split(digit_images, labels)

x_train = torch.tensor(x_train, dtype=torch.float32)
t_train = torch.tensor(t_train, dtype=torch.int64)
x_test = torch.tensor(x_test, dtype=torch.float32)
t_test = torch.tensor(t_test, dtype=torch.int64)


net = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10),
)
print(net)

loss_fnc = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.01)

record_loss_train: List[float] = []
record_loss_test: List[float] = []

epochs = 1000

for i in range(epochs):
    optimizer.zero_grad()

    y_train = net(x_train)
    y_test = net(x_test)

    loss_train = loss_fnc(y_train, t_train)
    loss_test = loss_fnc(y_test, t_test)
    record_loss_train.append(loss_train.item())
    record_loss_test.append(loss_test.item())

    loss_train.backward()

    optimizer.step()

    if i % 100 == 0:
        print(f"Epoch: {i}, Loss_Train: {loss_train}, Loss_Test: {loss_test}")

plt.plot(range(len(record_loss_train)), record_loss_train, label="Train")
plt.plot(range(len(record_loss_test)), record_loss_test, label="Test")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

y_test = net(x_test)
count = (y_test.argmax(1) == t_test).sum().item()
print(f"correct rate: {count/len(y_test)*100}%")

img_id = 0
x_pred = digit_images[img_id]
print("type", type(x_pred))
image = x_pred.reshape(8, 8)
plt.imshow(image, cmap="Greys_r")
plt.show()

x_pred = torch.tensor(x_pred, dtype=torch.float32)
y_pred = net(x_pred)
print(f"answer: {labels[img_id]}, prediction: {y_pred.argmax().item()}")
