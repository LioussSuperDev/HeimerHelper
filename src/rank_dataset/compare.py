import torch
import torch.nn as nn
import torch.nn.functional as F
import model_architectures
import dataset_small
import dataset_medium
import dataset_big
import sys
import numpy as np
import torch_directml
from dataset_utils import _handle_match
sys.path.insert(0, '..')
import gatherer_utils
import json
import os
sys.path.remove('..')

matches = []
matches2 = []
files = ["last_live_game.json","last_played_game.json"]


#loading the game
for file in files:
    with open(file,"r") as f:
        matches.append(json.load(f))
    

device = torch_directml.device()

for match in matches:
    matches2.append(dataset_medium.json_to_numpy(match))

for i in range(matches2[0].shape[0]):
    if matches2[0][i].item() != matches2[1][i].item():
        print(i," - ",matches2[0][i].item(),matches2[1][i].item())