import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy


a = torch.Tensor([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1],[1, 1, 1, 1]])
b = torch.Tensor([[-1, -1, -1, -2], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
print(a.numpy())

def custom_loss(prediction, label):
    loss = torch.abs(label-prediction)
    avg_throttle_loss = numpy.mean(loss.numpy()[:, 0])
    sum_throttle_loss = sum(loss.numpy()[:, 0])
    total_loss = torch.sum(loss)
    return loss, total_loss, avg_throttle_loss, sum_throttle_loss

J = custom_loss(a, b)

print(J)
print(J[1])
print(J[2])
print(J[3])

c="([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1],[1, 1, 1, 1]])"
print(c.strip("()"))
