import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from carball.json_parser.game import Game
from carball.analysis.analysis_manager import AnalysisManager
from carball.controls.controls import ControlsCreator
import carball
import os
import json
'######################################################'
'1. Dataloader'
'######################################################'
'''needs to extract replay tick packet data as features
and controller outputs as labels with "Carball"
into a feature matrix and label vector to input into the model'''

raw_replays= []

src_path = os.path.abspath("") + "\\"
#print("Test", src_path)
replay_path = os.path.abspath("raw_replays") + "\\"
#print("TEST", replay_path)

replay_files = os.listdir(replay_path)
#print(replay_files)

for i in replay_files:
    replay_file = replay_path + i
    raw_replays.append(replay_file)

#print(raw_replays)

for replay_file in raw_replays:
    json_data = carball.decompile_replay(replay_file)

    game = Game()
    game.initialize(loaded_json = json_data)

    analysis = AnalysisManager(game)
    analysis.create_analysis()
    df = analysis.get_data_frame()

    """
    CREATING FEATURE DATA as X_tensor and loading it into featuresc.csv
    """

    player_1 = df.columns[0][0]
    print(player_1)
    player_2 = df.columns[22][0]
    print(player_2)

    features = df[[
    (player_1, "pos_x"), (player_1, "pos_y"), (player_1, "pos_z"), (player_1, "rot_x"), (player_1, "rot_y"), (player_1, "rot_z"), (player_1, "vel_x"),
    (player_1, "vel_y"), (player_1, "vel_z"), (player_1, "ang_vel_x"), (player_1, "ang_vel_y"), (player_1, "ang_vel_z"), (player_1, "boost"),

    (player_2, "pos_x"), (player_2, "pos_y"), (player_2, "pos_z"), (player_2, "rot_x"), (player_2, "rot_y"), (player_2, "rot_z"), (player_2, "vel_x"),
    (player_2, "vel_y"), (player_2, "vel_z"), (player_2, "ang_vel_x"), (player_2, "ang_vel_y"), (player_2, "ang_vel_z"), (player_2, "boost"),

    ("ball", "pos_x"), ("ball", "pos_y"), ("ball", "pos_z"), ("ball", "rot_x"), ("ball", "rot_y"), ("ball", "rot_z"), ("ball", "vel_x"),
    ("ball", "vel_y"), ("ball", "vel_z"), ("ball", "ang_vel_x"), ("ball", "ang_vel_y"), ("ball", "ang_vel_z")]]

    #replaces NaN with zeroes
    features.fillna(0, inplace = True)

    #rescales each player's boost amount from 0-255 to 0-100
    features[(player_1, "boost")] = features[(player_1, "boost")] / 2.55
    features[(player_2, "boost")] = features[(player_2, "boost")] / 2.55

    #print("FEATURES\n",features, features.shape)

    #if the first file, keep header and dont append to csv file
    #after the first file, don't keep header and append to csv file at the end of the last replay data
    if raw_replays.index(replay_file) == 0:
        features.to_csv(src_path + r"features.csv", header = True)
    else:
        features.to_csv(src_path + r"features.csv", header = False, mode = 'a')

    """
    CREATING LABEL DATA as Y_tensor and loading it into labels.csv
    """
    controls = ControlsCreator()
    outputs = controls.get_controls(game)
    player = game.players[0]
    labels = player.controls

    #replacing boolean with 1s and 0s
    labels.replace(to_replace = True, value = 1, inplace = True)
    labels.replace(to_replace = False, value = 0, inplace = True)

    #replaces NaN with zeroes
    labels.fillna(0, inplace = True)

    #print(labels)

    if raw_replays.index(replay_file) == 0:
        labels.to_csv(src_path + r"labels.csv", header = True)
    else:
        labels.to_csv(src_path + r"labels.csv", header = False, mode = 'a')

n_in = 38
n_mid = int((n_in + 8)/2)
n_out = 8

'######################################################'
'2. Model'
'######################################################'

class AlphaSlow(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(AlphaSlow, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_out)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        outputs = self.sigmoid(self.fc1(inputs))
        outputs = self.tanh(self.fc2(outputs))
        return outputs

model = AlphaSlow(n_in, n_mid, n_out)

'######################################################'
'3. Loss Function'
'######################################################'
loss = nn.HingeEmbeddingLoss()
learning_rate = 1e-2

'######################################################'
'4. Optimization Algorithm (SGD)'
'######################################################'
opt = optim.SGD(model.parameters(), lr=learning_rate)

'######################################################'
'5. Minibatch iteration'
'######################################################'
epochs = 10

# an empty list to collect the training loss (used for graphing)
J_train_all = []
# an empty list to collect the testing loss (used for graphing)
J_test_all = []

#for every one of the epochs:
for i in range(epochs):
    #iterate through each frame sample (of all replays which are concatenated in the X tensor)
    for index,sample in features.iterrows():
        X_tensor = torch.tensor(sample.values).float()
        Y_tensor = torch.tensor(labels.loc[index, :].values).float()
        model.train()
        opt.zero_grad()
        y_hat = model(X_tensor)
        J = loss(y_hat, Y_tensor)
        J.backward()
        opt.step()

        #printing the loss every iteration
        #if index %9000 == 0:
            #print('SAMPLE: ', index)
            #print("LOSS: ", J)

parameter_file = (src_path + r"parameters.txt", "w+")
for param in model.parameters():
  print(param.data)
  parameter_file.write(str(param.data))
print("TRAINING COMPLETE YIPPEY")
