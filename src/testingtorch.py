import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

print(os.path.abspath(r"1.replay"))

print(os.listdir(r"C:\Users\ollie\AppData\Local\RLBotGUIX\MyBots\AlphaSlow\src"))
import createTrainingData
print(os.path.exists(r"C:\Users\ollie\AppData\Local\RLBotGUIX\MyBots\AlphaSlow\src\json_replays"))
print("OK")
print(os.path.isfile(r"C:\Users\ollie\AppData\Local\RLBotGUIX\MyBots\AlphaSlow\src\json_replays"))
print(os.access(r'C:\Users\ollie\AppData\Local\RLBotGUIX\MyBots\AlphaSlow\src\json_replays', os.R_OK))
print(os.access(r'C:\Users\ollie\AppData\Local\RLBotGUIX\MyBots\AlphaSlow\src\json_replays', os.W_OK))
print(os.access(r'C:\Users\ollie\AppData\Local\RLBotGUIX\MyBots\AlphaSlow\src\json_replays', os.X_OK))
print(os.access(r'C:\Users\ollie\AppData\Local\RLBotGUIX\MyBots\AlphaSlow\src\json_replays', os.F_OK))
print(os.stat(r'C:\Users\ollie\AppData\Local\RLBotGUIX\MyBots\AlphaSlow\src\json_replays'))
#open(r"C:\Users\ollie\AppData\Local\RLBotGUIX\MyBots\AlphaSlow\src\json_replays")
#open(r"C:\Users\ollie\AppData\Local\RLBotGUIX\MyBots\AlphaSlow\src\json_replays",encoding='utf-8', errors='ignore')

print("TESTING LOCAL PATH")
print(os.path.exists(r"\Users\ollie\AppData\Local\RLBotGUIX\MyBots\AlphaSlow\src\json_replays"))
print("OK")
print(os.path.isfile(r"json_replays"))
print(os.access(r'json_replays', os.R_OK))
print(os.access(r'json_replays', os.W_OK))
print(os.access(r'json_replays', os.X_OK))
print(os.access(r'json_replays', os.F_OK))

print('TESTING ABS PATH')
print(os.path.abspath(r'Users\ollie\AppData\Local\RLBotGUIX\MyBots\AlphaSlow\src\replays_replays'))
