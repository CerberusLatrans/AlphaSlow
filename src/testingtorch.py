
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


tup_in_list=[("me", 1), ("you", 2), ("him",3)]

print(tup_in_list[0][0])

list = [0.75, 0.73, 98, 4.5, 23.1, 0.69, 0.93]

#VEC3  Notes
#A= Vec3

#A = no error
#Vec3(A) = could not convert string to float
#A[0] = Vector 3 object not subscriptable
#list(A) = Vector 3 object is not iterable
#tuple(A) = Vector 3 object is not iterable
#A['x'] = vector 3 object is not subscriptable
#A.x = NO ERROR :)

#rotator and vec3 are two different types of objects

v = torch.empty(10,10)
print(v)
nn.init.uniform_(v)
print(v)

print(torch.Tensor(5))
print(len(torch.Tensor(5)))

pear = 3
print("apple" + str(1+pear))


for a in range(10):
    print(a)
