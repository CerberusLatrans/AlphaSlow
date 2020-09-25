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
import math

'######################################################'
'1. Dataloader'
'######################################################'
'''needs to extract replay tick packet data as features
and controller outputs as labels with "Carball"
into a feature matrix and label vector to input into the model
then uploads features and labels to respecitve csv files for model_training.py to use'''

raw_replays= []

src_path = os.path.abspath(__file__)[:-13]
#print("Test", src_path)
replay_path = src_path + "raw_replays" + "\\"
#print("TEST", replay_path)

replay_files = os.listdir(replay_path)
#print(replay_files)

features_file = (src_path + r"features.csv")
out_features = open(features_file, "a+")
out_features.truncate(0)

labels_file = (src_path + r"labels.csv")
out_labels = open(labels_file, "a+")
out_labels.truncate(0)

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
    CREATING FEATURE DATA as X_tensor
    """

    player_1 = df.columns[0][0]
    print("PLAYER 1: ", player_1)

    player_2 = df.columns[23][0]
    print("PLAYER 2: ", player_2)

    features = df[[
    (player_1, "pos_x"), (player_1, "pos_y"), (player_1, "pos_z"), (player_1, "rot_x"), (player_1, "rot_y"), (player_1, "rot_z"), (player_1, "vel_x"),
    (player_1, "vel_y"), (player_1, "vel_z"), (player_1, "ang_vel_x"), (player_1, "ang_vel_y"), (player_1, "ang_vel_z"), (player_1, "boost"),

    (player_2, "pos_x"), (player_2, "pos_y"), (player_2, "pos_z"), (player_2, "rot_x"), (player_2, "rot_y"), (player_2, "rot_z"), (player_2, "vel_x"),
    (player_2, "vel_y"), (player_2, "vel_z"), (player_2, "ang_vel_x"), (player_2, "ang_vel_y"), (player_2, "ang_vel_z"), (player_2, "boost"),

    ("ball", "pos_x"), ("ball", "pos_y"), ("ball", "pos_z"), ("ball", "rot_x"), ("ball", "rot_y"), ("ball", "rot_z"), ("ball", "vel_x"),
    ("ball", "vel_y"), ("ball", "vel_z"), ("ball", "ang_vel_x"), ("ball", "ang_vel_y"), ("ball", "ang_vel_z")]]

    #replaces NaN with zeroes
    features = features.fillna(0)

    #print("FEATURES\n",features, features.shape)
    """
    ########################################
    NORMALIZATION (makes all feature values between 0 and 1)
    ########################################
    """
    #for feature in features.columns:
        #print(feature, "MAX: ", max(features[feature].values), "MIN: ", min(features[feature].values))

    def norm(dataset, feature, min, max):
        for value in dataset.loc[:, feature].values:
            new_value = (value - min) /  (max - min)
            dataset = dataset.replace(value, new_value)
        return dataset

    norm(features, (player_1, "pos_x"), -4096, 4096)
    norm(features, (player_1, "pos_y"), -4096, 4096)
    norm(features, (player_1, "pos_z"), 0, 2044)
    norm(features, (player_1, "rot_x"), -math.pi/2, math.pi/2)
    norm(features, (player_1, "rot_y"), -math.pi, math.pi)
    norm(features, (player_1, "rot_z"), -math.pi, math.pi)
    norm(features, (player_1, "vel_x"), -23000, 23000)
    norm(features, (player_1, "vel_y"), -23000, 23000)
    norm(features, (player_1, "vel_z"), -23000, 23000)
    norm(features, (player_1, "ang_vel_x"), -5500, 5500)
    norm(features, (player_1, "ang_vel_y"), -5500, 5500)
    norm(features, (player_1, "ang_vel_z"), -5500, 5500)
    norm(features, (player_1, "boost"), 0, 255)

    norm(features, (player_2, "pos_x"), -4096, 4096)
    norm(features, (player_2, "pos_y"), -4096, 4096)
    norm(features, (player_2, "pos_z"), 0, 2044)
    norm(features, (player_2, "rot_x"), -math.pi/2, math.pi/2)
    norm(features, (player_2, "rot_y"), -math.pi, math.pi)
    norm(features, (player_2, "rot_z"), -math.pi, math.pi)
    norm(features, (player_2, "vel_x"), -23000, 23000)
    norm(features, (player_2, "vel_y"), -23000, 23000)
    norm(features, (player_2, "vel_z"), -23000, 23000)
    norm(features, (player_2, "ang_vel_x"), -5500, 5500)
    norm(features, (player_2, "ang_vel_y"), -5500, 5500)
    norm(features, (player_2, "ang_vel_z"), -5500, 5500)
    norm(features, (player_2, "boost"), 0, 255)

    norm(features, ("ball", "pos_x"), -4096, 4096)
    norm(features, ("ball", "pos_y"), -4096, 4096)
    norm(features, ("ball", "pos_z"), 0, 2044)
    norm(features, ("ball", "rot_x"), -math.pi/2, math.pi/2)
    norm(features, ("ball", "rot_y"), -math.pi, math.pi)
    norm(features, ("ball", "rot_z"), -math.pi, math.pi)
    norm(features, ("ball", "vel_x"), -60000, 60000)
    norm(features, ("ball", "vel_y"), -60000, 60000)
    norm(features, ("ball", "vel_z"), -60000, 60000)
    norm(features, ("ball", "ang_vel_x"), -6000, 6000)
    norm(features, ("ball", "ang_vel_y"), -6000, 6000)
    norm(features, ("ball", "ang_vel_z"), -6000, 6000)

    ##for feature in features.columns:
        ##print(feature, "MAX: ", max(features[feature].values), "MIN: ", min(features[feature].values))
    """
    LOADING INTO features.csv FILE
    """

    #if the first file, keep header and dont append to csv file
    #after the first file, don't keep header and append to csv file at the end of the last replay data
    if raw_replays.index(replay_file) == 0:
        features.to_csv(src_path + r"features.csv", header = True, index = False)
    else:
        features.to_csv(src_path + r"features.csv", header = False, index = False, mode = 'a')


    """
    CREATING LABEL DATA as Y_tensor and loading it into labels.csv
    """
    controls = ControlsCreator()
    controls.get_controls(game)
    player = game.players[0]
    labels = player.controls

    #replacing boolean with 1s and -1s
    labels.replace(to_replace = True, value = 1, inplace = True)
    labels.replace(to_replace = False, value = -1, inplace = True)

    #replaces NaN with zeroes
    labels.fillna(0, inplace = True)

    #for label in labels.columns:
        #print(label, "MAX: ", max(labels[label].values), "MIN: ", min(labels[label].values))
    #print(labels)

    if raw_replays.index(replay_file) == 0:
        labels.to_csv(src_path + r"labels.csv", header = True, index = False)
    else:
        labels.to_csv(src_path + r"labels.csv", header = False, index = False, mode = 'a')
