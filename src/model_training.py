import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from carball.json_parser.game import Game
from carball.analysis.analysis_manager import AnalysisManager
from carball.controls.controls import ControlsCreator
import carball
import os
import json

src_path = os.path.abspath(__file__)[:-17]

torch.autograd.set_detect_anomaly(True)

'######################################################'
'2. Model'
'######################################################'

class AlphaSlow(nn.Module):
    def __init__(self, n_in, n_out):
        super(AlphaSlow, self).__init__()
        self.fc1 = nn.Linear(n_in, 320, bias = True)
        self.fc2 = nn.Linear(320, 160, bias = True)
        self.fc3 = nn.Linear(160, 80, bias = True)
        self.fc4 = nn.Linear(80, 80, bias = True)
        self.fc5 = nn.Linear(80, 40, bias = True)
        self.fc6 = nn.Linear(40, 40, bias = True)
        self.fc7 = nn.Linear(40, 20, bias = True)
        self.fc8 = nn.Linear(20, 20, bias = True)
        self.fc9 = nn.Linear(20, n_out, bias = True)

        self.sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()
        self.ReLU = nn.ReLU()

    def forward(self, inputs):
        outputs = self.ReLU(self.fc1(inputs))
        outputs = self.ReLU(self.fc2(outputs))
        outputs = self.ReLU(self.fc3(outputs))
        outputs = self.ReLU(self.fc4(outputs))
        outputs = self.ReLU(self.fc5(outputs))
        outputs = self.ReLU(self.fc6(outputs))
        outputs = self.ReLU(self.fc7(outputs))
        outputs = self.ReLU(self.fc8(outputs))
        outputs = self.Tanh(self.fc9(outputs))

        return outputs
n_in = 38
n_out = 8

model = AlphaSlow(n_in, n_out)

'######################################################'
'3. Loss Function'
'######################################################'
def custom_loss(prediction, label):
    loss = torch.abs(label-prediction)
    avg_total_loss = numpy.mean(loss.detach().numpy())
    tensor_avg_total_loss = torch.mean(loss)

    avg_throttle_loss = numpy.mean(loss.detach().numpy()[:, 0])
    avg_steer_loss = numpy.mean(loss.detach().numpy()[:, 1])
    avg_pitch_loss = numpy.mean(loss.detach().numpy()[:, 2])
    avg_yaw_loss = numpy.mean(loss.detach().numpy()[:, 3])
    avg_roll_loss = numpy.mean(loss.detach().numpy()[:, 4])
    avg_jump_loss = numpy.mean(loss.detach().numpy()[:, 5])
    avg_boost_loss = numpy.mean(loss.detach().numpy()[:, 6])
    avg_powerslide_loss = numpy.mean(loss.detach().numpy()[:, 7])

    return loss, avg_total_loss, tensor_avg_total_loss, avg_throttle_loss, avg_steer_loss, avg_pitch_loss, avg_yaw_loss, avg_roll_loss, avg_jump_loss, avg_boost_loss, avg_powerslide_loss

"################################################################################################################################################################################################################################################################################################"
learning_rate = 1e-1
momentum = 0.5
"################################################################################################################################################################################################################################################################################################"


'######################################################'
'4. Optimization Algorithm (SGD)'
'######################################################'
optimizer = "MBGD"
opt = optim.SGD(model.parameters(), lr=learning_rate, momentum = momentum, nesterov = False)
#opt = optim.Adam(model.parameters(), lr=learning_rate)

'######################################################'
'5. Iteration'
'######################################################'

class CustomDataset(Dataset):
    def __init__(self, inp_csv, label_csv):
        feature_data = pd.read_csv(inp_csv, delimiter = ",", dtype=numpy.float32, skiprows=2)
        label_data = pd.read_csv(label_csv, delimiter = ",", dtype=numpy.float32, skiprows=1)
        self.X_data=torch.Tensor(feature_data.values)
        self.y_data=torch.Tensor(label_data.values)
        self.n_samples = feature_data.shape[0]

    #returns the index of each sample
    def __getitem__(self, index):
        X = self.X_data[index]
        Y = self.y_data[index]
        sample = {'X': X, 'Y': Y}
        return sample

    #returns the size of each sample
    def __len__(self):
      return self.n_samples

dataset = CustomDataset(src_path + "features.csv", src_path + "labels.csv")
print(dataset)
"################################################################################################################################################################################################################################################################################################"
dataloader_train= DataLoader(dataset=dataset, batch_size=10000, shuffle=True)
print(dataloader_train)

epochs = 10000
"################################################################################################################################################################################################################################################################################################"

# an empty list to collect the training loss (used for graphing)
J_train_all = []
J_train_avg = []

J_throttle_avg = []
J_steer_avg = []
J_pitch_avg = []
J_yaw_avg = []
J_roll_avg = []
J_jump_avg = []
J_boost_avg = []
J_powerslide_avg = []

#for every one of the epochs:
for i in range(epochs):
    #iterate through each frame sample (of all replays which are concatenated in the X tensor)
    for index, sample in enumerate(dataloader_train):
        X = sample["X"]
        #print("X: ", X, X.size())
        Y = sample["Y"]
        #print("Y: ", Y, Y.size())

        model.train()
        opt.zero_grad()
        y_hat = model(X)

        #print("Y_HAT: ", y_hat, yhat.size())

        J_all = custom_loss(y_hat, Y)
        J_avg = J_all[2]
        J_avg.backward()
        opt.step()

        #adding loss to list to be graphed every iteration
        J_train_avg.append(J_all[1])

        J_throttle_avg.append(J_all[3])
        J_steer_avg.append(J_all[4])
        J_pitch_avg.append(J_all[5])
        J_yaw_avg.append(J_all[6])
        J_roll_avg.append(J_all[7])
        J_jump_avg.append(J_all[8])
        J_boost_avg.append(J_all[9])
        J_powerslide_avg.append(J_all[10])


        if index % 1 == 0:
            print('BATCH: ', index+1, "EPOCH: ", i+1)
            print(f"AVERAGE LOSS (BATCH SIZE = {dataloader_train.batch_size}): ", J_all[1])

parameter_file = (src_path + r"parameters.txt")
out = open(parameter_file, "a+")
out.truncate(0)

torch.set_printoptions(threshold=1000000)

for param in model.parameters():
    print(param.data)
    out.write(str(param.data))

out.close()
print("TRAINING COMPLETE YIPPEY")

"""
#######################################
GRAPHING LOSS OVER ITERATION
#######################################
"""

os.mkdir(src_path + r"loss_graphs\\" + f"momentum = 0.5, Normalized, ReLU, 10-layer [38,320, 160, 80, 80, 40, 40, 20, 20, 8], {optimizer}, lr={learning_rate}, epochs={epochs}, batch_size={dataloader_train.batch_size}, samples={dataset.n_samples+1}")
loss_folder_path = src_path + r"loss_graphs\\" + f"momentum = 0.5, Normalized, ReLU, 10-layer [38,320, 160, 80, 80, 40, 40, 20, 20, 8], {optimizer}, lr={learning_rate}, epochs={epochs}, batch_size={dataloader_train.batch_size}, samples={dataset.n_samples+1}"
print(loss_folder_path)

plt.plot(J_train_avg)
plt.title(f"{optimizer}, lr={learning_rate}, epochs={epochs}, batch_size={dataloader_train.batch_size}, samples={dataset.n_samples+1}", fontdict = {'fontsize' : 10})
plt.ylabel('AVG LOSS')
plt.xlabel('# OF BATCH ITERATIONS (batches/epoch * epochs)')
plt.yticks(numpy.arange(0, 1, step=0.1))
plt.savefig(loss_folder_path + r'\avg_loss.png')
plt.show()

plt.plot(J_throttle_avg)
plt.title(f"{optimizer}, lr={learning_rate}, epochs={epochs}, batch_size={dataloader_train.batch_size}, samples={dataset.n_samples+1}", fontdict = {'fontsize' : 10})
plt.ylabel('AVG THROTTLE LOSS')
plt.xlabel('# OF BATCH ITERATIONS (batches/epoch * epochs)')
plt.yticks(numpy.arange(0, 1, step=0.1))
plt.savefig(loss_folder_path + r'\avg_throttle_loss.png')
plt.show()

plt.plot(J_steer_avg)
plt.title(f"{optimizer}, lr={learning_rate}, epochs={epochs}, batch_size={dataloader_train.batch_size}, samples={dataset.n_samples+1}", fontdict = {'fontsize' : 10})
plt.ylabel('AVG STEER LOSS')
plt.xlabel('# OF BATCH ITERATIONS (batches/epoch * epochs)')
plt.yticks(numpy.arange(0, 1, step=0.1))
plt.savefig(loss_folder_path + r'\avg_steer_loss.png')
plt.show()

plt.plot(J_pitch_avg)
plt.title(f"{optimizer}, lr={learning_rate}, epochs={epochs}, batch_size={dataloader_train.batch_size}, samples={dataset.n_samples+1}", fontdict = {'fontsize' : 10})
plt.ylabel('AVG PITCH LOSS')
plt.xlabel('# OF BATCH ITERATIONS (batches/epoch * epochs)')
plt.yticks(numpy.arange(0, 1, step=0.1))
plt.savefig(loss_folder_path + r'\avg_pitch_loss.png')
plt.show()

plt.plot(J_yaw_avg)
plt.title(f"{optimizer}, lr={learning_rate}, epochs={epochs}, batch_size={dataloader_train.batch_size}, samples={dataset.n_samples+1}", fontdict = {'fontsize' : 10})
plt.ylabel('AVG YAW LOSS')
plt.xlabel('# OF BATCH ITERATIONS (batches/epoch * epochs)')
plt.yticks(numpy.arange(0, 1, step=0.1))
plt.savefig(loss_folder_path + r'\avg_yaw_loss.png')
plt.show()

plt.plot(J_roll_avg)
plt.title(f"{optimizer}, lr={learning_rate}, epochs={epochs}, batch_size={dataloader_train.batch_size}, samples={dataset.n_samples+1}", fontdict = {'fontsize' : 10})
plt.ylabel('AVG ROLL LOSS')
plt.xlabel('# OF BATCH ITERATIONS (batches/epoch * epochs)')
plt.yticks(numpy.arange(0, 1, step=0.1))
plt.savefig(loss_folder_path + r'\avg_roll_loss.png')
plt.show()

plt.plot(J_jump_avg)
plt.title(f"{optimizer}, lr={learning_rate}, epochs={epochs}, batch_size={dataloader_train.batch_size}, samples={dataset.n_samples+1}", fontdict = {'fontsize' : 10})
plt.ylabel('AVG JUMP LOSS')
plt.xlabel('# OF BATCH ITERATIONS (batches/epoch * epochs)')
plt.yticks(numpy.arange(0, 1, step=0.1))
plt.savefig(loss_folder_path + r'\avg_jump_loss.png')
plt.show()

plt.plot(J_boost_avg)
plt.title(f"{optimizer}, lr={learning_rate}, epochs={epochs}, batch_size={dataloader_train.batch_size}, samples={dataset.n_samples+1}", fontdict = {'fontsize' : 10})
plt.ylabel('AVG BOOST LOSS')
plt.xlabel('# OF BATCH ITERATIONS (batches/epoch * epochs)')
plt.yticks(numpy.arange(0, 1, step=0.1))
plt.savefig(loss_folder_path + r'\avg_boost_loss.png')
plt.show()

plt.plot(J_powerslide_avg)
plt.title(f"{optimizer}, lr={learning_rate}, epochs={epochs}, batch_size={dataloader_train.batch_size}, samples={dataset.n_samples+1}", fontdict = {'fontsize' : 10})
plt.ylabel('AVG POWERSLIDE LOSS')
plt.xlabel('# OF BATCH ITERATIONS (batches/epoch * epochs)')
plt.yticks(numpy.arange(0, 1, step=0.1))
plt.savefig(loss_folder_path + r'\avg_powerslide_loss.png')
plt.show()

quit()
