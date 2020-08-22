import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os


import re
src_path = os.path.abspath("") + "\\"
print(src_path)
with open(src_path + "parameters.txt", "r") as file:
    parameters = file.read().replace('\n', '')
    print(parameters)
    tensor = re.findall("\[\[.+?\]\]", parameters)
    print(len(tensor), tensor)
