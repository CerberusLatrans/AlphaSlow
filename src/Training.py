import math
import rlbot.utils.structures.game_data_struct as game_data_struct
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

'######################################################'
'1. Dataloader'
'######################################################'
'''needs to extract replay tick packet data as features with "Training Data Extractor"
and controller outputs as labels with "Carball"
into a feature matrix and label vector to input into the model'''
n_in = ??
n_mid = 10
n_out = 7

'######################################################'
'2. Model'
'######################################################'
class AS_model(nn.Module):
    controller =  SimpleControllerState()
    def __init__(self, name, team, index, n_in, n_out):
        self.index = index
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_out)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    def propagate(self, inputs):
        outputs = self.sigmoid(self.fc1(inputs))
        outputs = self.tanh(self.fc2(outputs))

    def output(self, output_vector):
        controller.throttle = output_vector[0]
        controller.steer = output_vector[1]
        controller.pitch = output_vector[2]
        controller.yaw = output_vector[3]
        controller.roll = output_vector[4]
        if output_vector[5] >= 0:
            controller.jump = True
        else:
            controller.jump = False
        if output_vector[6] >= 0:
            controller.boost = True
        else:
            controller.boost = False
        if output_vector[7] >= 0:
            controller.handbrake = True
        else:
            controller.handbrake = False
        return controller
model = AlphaSlow(n_input, n_output)
loss = nn.MSELoss(reduction='mean')
learning_rate = 1e-2
opt = optim.SGD(model.parameters(), lr=learning_rate)
epochs = 10
