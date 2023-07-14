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




#loading the game
sn = input("enter the summoner's name of one of the players in the game : ")
choix = int(input("please enter the number of games you wish to analyse in match history (0 for live game) : "))

live_game = False
if choix == 0:
    ma,mas,t = gatherer_utils.gather_live_game("euw1",sn)
    matches = [("live",ma,mas,t)]
    live_game = True
else:
    _,matches = gatherer_utils.load_player_data("euw1", sn, max_number_of_matches=choix)

print("match(es) loaded !")

sys.path.remove('..')

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
device = torch_directml.device()

print("loading models...")
model0 = model_architectures.MLP2(dataset_medium.get_datasize())
model0.load_state_dict(torch.load("models\\MLP2\\0.8034_l0.01_w0.0001_dsetmedium.state", map_location=device))
model0.eval()

model1 = model_architectures.MLP2(dataset_medium.get_datasize())
model1.load_state_dict(torch.load("models\\MLP2\\0.7844_l0.005_w0.0001_dsetmedium.state", map_location=device))
model1.eval()

model2 = model_architectures.MLP3(dataset_small.get_datasize())
model2.load_state_dict(torch.load("models\\MLP3\\0.7723_l0.005_w0.0001_dsetsmall.state", map_location=device))
model2.eval()

model3 = model_architectures.MLP3(dataset_small.get_datasize())
model3.load_state_dict(torch.load("models\\MLP3\\0.8035_l0.005_w0.0001_dsetsmall.state", map_location=device))
model3.eval()

for name,match,masteries,team_nb in matches:

    handled_match = _handle_match(match, masteries, 0, include_victory=False)

    proper_match1 = dataset_medium.json_to_numpy(handled_match)
    proper_match2 = dataset_small.json_to_numpy(handled_match)

    outputs = []
    outputs.append(model0(torch.tensor(proper_match1).unsqueeze(dim=0))[0].item())
    outputs.append(model1(torch.tensor(proper_match1).unsqueeze(dim=0))[0].item())
    outputs.append(model2(torch.tensor(proper_match2).unsqueeze(dim=0))[0].item())
    outputs.append(model3(torch.tensor(proper_match2).unsqueeze(dim=0))[0].item())

    print("=============")
    print("DOING MATCH",name)
    print("=============")
    print("")
    for idx,output in enumerate(outputs):
        print("---")
        print("Result of model",(idx+1))
        certitude = output
        
        gagnant = 1
        if certitude <= 0.5:
            certitude = (2*(0.5 - certitude))
            gagnant = 2
        else:
            certitude = (2*(certitude - 0.5))
        joueurGagnant = gagnant == team_nb
        certitude = certitude**(1/2)
        print("prédiction originale :",round(output,4))
        print("joueur gagnant :",joueurGagnant)
        print("certitude :",str(round(certitude*100,2))+"%")
        print("---")
    print("")