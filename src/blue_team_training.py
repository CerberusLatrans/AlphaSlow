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

from carball.json_parser.game import Game
from carball.analysis.analysis_manager import AnalysisManager
from carball.controls.controls import ControlsCreator
import carball
import os
import json
'######################################################'
'1. Dataloader'
'######################################################'
'''needs to extract replay tick packet data as features with "Training Data Extractor"
and controller outputs as labels with "Carball"
into a feature matrix and label vector to input into the model'''

raw_replays= []

replay_path = r"C:\Users\ollie\AppData\Local\RLBotGUIX\MyBots\AlphaSlow\src\raw_replays""\\"

replay_files = os.listdir(replay_path)
print(replay_files)

for i in replay_files:
    replay_file = replay_path + str(replay_files.index(i)+1) + ".replay"
    raw_replays.append(replay_file)

print(raw_replays)

for replay_file in raw_replays:
    json_data = carball.decompile_replay(replay_file)
    print("JSON:", json_data)

    game = Game()
    game.initialize(loaded_json = json_data)

    analysis = AnalysisManager(game)
    analysis.create_analysis()
    df = analysis.get_data_frame()



    print(df.loc[:, ("ApparentlyJack", "pos_x")])
    print(df.loc[:, ("Oski", "pos_x")])
    controls = ControlsCreator()
    outputs = controls.get_controls(game)
    print(player.controls)

    #if the first file, keep header and dont append to csv file
    #after the first file, don't keep header and append to csv file at the end of the last replay data
    if raw_replays.index(replay_file) == 0:
        df.to_csv(r"C:\Users\ollie\AppData\Local\RLBotGUIX\MyBots\AlphaSlow\src\game_data.csv", header = True)
    else:
        df.to_csv(r"C:\Users\ollie\AppData\Local\RLBotGUIX\MyBots\AlphaSlow\src\game_data.csv", header = False, mode = 'a')


n_features = 36
n_mid = (n_features + 8)/2
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

model = AlphaSlow(n_features, n_mid, n_out)

'######################################################'
'3. Loss Function'
'######################################################'
loss = nn.CrossEntropyLoss()
learning_rate = 1e-2

'######################################################'
'4. Optimization Algorithm (SGD)'
'######################################################'
opt = optim.SGD(model.parameters(), lr=learning_rate)

'######################################################'
'5. Minibatch iteration'
'######################################################'
epochs = 10000

# an empty list to collect the training loss (used for graphing)
J_train_all = []
# an empty list to collect the testing loss (used for graphing)
J_test_all = []

#for every one of the 10000 epochs:
for i in range(epochs):
    #since total training samples is 800 and training batch size is 50
    #it will have 16 batches to iterate through per epoch
    #for each batch in the training dataset:
    for _, sample in enumerate(dataloader_train):
        # set model to training mode
        model.train()
        # clear gradients information from the previous iteration
        opt.zero_grad()
        # read out features and labels from the mini-batch
        x = sample['X']
        y = sample['y']

        # predict the labels using the model
        y_hat = model(x)
        # compute the loss
        J = loss(y_hat, y)
        # compute the gradients
        J.backward()
        # update the parameters using the optimizer
        opt.step()

    # To check the training loss every 100 epochs:
    if i % 100 == 0:
        # set the model to evaluation mode
        model.eval()
        # we don't really need to compute the gradients here
        # so temporally turn it off to accelerate the computation
        #since total training samples is 800 and training batch size is 800
            #it will have 1 batch to iterate through per epoch
            #for each batch in the testing dataset (only 1):
        for _, sample in enumerate(dataloader_train_full_dataset):
          with torch.no_grad():
            J_train = loss(model(sample['X']), sample['y'])
            J_train_all.append(J_train)
            #print (J_train.numpy())
            # print() is ugly and slows down the code.
            # Pro tips: Google the tqdm library and learn how to show a nicer training progress bar!

    #to check the testing loss every 100 epochs:
    if i % 100 == 0:
        model.eval()
        #since total testing samples is 200 and training batch size is 200
            #it will have 1 batch to iterate through per epoch
            #for each batch in the testing dataset (only 1):
        for _, sample in enumerate(dataloader_test):
          with torch.no_grad():
            J_test = loss(model(sample['X']), sample['y'])
            J_test_all.append(J_test)
            print (J_test.numpy())
            print("OUTPUT SHAPE", y_hat.shape, type(y_hat))
