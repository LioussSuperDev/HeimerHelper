import torch
import model_architectures
import dataset_fullgame
import sys
# import torch_directml
from dataset_utils import _handle_match
sys.path.insert(0, '..')
import gatherer_utils
import json
import os

matches = None


def use_models(outputs, prediction_mode, team_nb=None):
    if prediction_mode == "d":
        print("=============")
        print("DOING MATCH",name)
        print("=============")
        print("")

    nb_T = 0
    cert_T = 0
    nb_F = 0
    cert_F = 0
    for idx,output in enumerate(outputs):

        if prediction_mode == "d":
            print("Result of model",(idx+1))


        certitude = output
        
        gagnant = 1
        if certitude <= 0.5:
            certitude = ((abs(0.5-certitude))+0.5)
            gagnant = 2
        else:
            certitude = ((abs(0.5-certitude))+0.5)
        if team_nb != None:
            joueurGagnant = gagnant == team_nb
        else:
            joueurGagnant = "UNKNOWN"
        certitude = certitude**(1/2)

        if gagnant == team_nb:
            nb_T += 1
            cert_T += certitude
        else:
            nb_F += 1
            cert_F += certitude


        if prediction_mode == "d":
            print("joueur gagnant :",joueurGagnant,str(round(certitude*100,3))+"% (winning team : "+str(gagnant)+")")

    if prediction_mode == "d":
        print("")

    if nb_T > nb_F:
        return "W"
    elif nb_T < nb_F:
        return "L"
    else:
        if cert_T > cert_T:
            return "W"
        else:
            return "L"







#loading the game
sn = input("enter the summoner's name of one of the players in the game : ")
if sn == "":
    sn = "sąf"

if os.path.isfile(sn):
    with open(sn,'r') as f:
        name,handled_match = "local_file",json.load(f)
    prediction_mode = "d"
else:
    choix = int(input("please enter the number of games you wish to analyse in match history (0 for live game) : "))
    prediction_mode = input("press d for detailed predictions, else press enter : ")

    live_game = False
    if choix == 0:
        ma,pmas,mas,t = gatherer_utils.gather_live_game("euw1",sn)
        matches = [("live",ma,pmas,mas,t)]
        live_game = True
    else:
        _,matches = gatherer_utils.load_player_data("euw1", sn, max_number_of_matches=choix, small_verbose=True)

print("match(es) loaded !")

sys.path.remove('..')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
# device = torch_directml.device()

print("loading models...")
model0 = model_architectures.MLP2(dataset_fullgame.get_datasize())
model0.load_state_dict(torch.load("models\\dataset_fullgame\\MLP2\\0.5111_l0.001_w0.0005_dsetdataset_fullgame.state", map_location=device))
model0.eval()

# model1 = model_architectures.MLP2(dataset_fullgame.get_datasize())
# model1.load_state_dict(torch.load("models\\MLP2\\0.7905_l0.005_w0.001_dsetsmall.state", map_location=device))
# model1.eval()

# model2 = model_architectures.MLP2(dataset_fullgame.get_datasize())
# model2.load_state_dict(torch.load("models\\MLP2\\0.811_l0.005_w0.001_dsetmedium.state", map_location=device))
# model2.eval()

# model3 = model_architectures.MLP2(dataset_fullgame.get_datasize())
# model3.load_state_dict(torch.load("models\\MLP2\\0.8082_l0.005_w0.001_dsetmedium.state", map_location=device))
# model3.eval()

predictions = []


if matches == None:
    outputs = []
    proper_match1 = dataset_fullgame.json_to_numpy(handled_match)
    # proper_match2 = dataset_fullgame.json_to_numpy(handled_match)
    # proper_match3 = dataset_fullgame.json_to_numpy(handled_match)
    outputs.append(model0(torch.tensor(proper_match1).unsqueeze(dim=0))[0].item())
    # outputs.append(model1(torch.tensor(proper_match3).unsqueeze(dim=0))[0].item())
    # outputs.append(model2(torch.tensor(proper_match1).unsqueeze(dim=0))[0].item())
    # outputs.append(model3(torch.tensor(proper_match1).unsqueeze(dim=0))[0].item())
    predictions.append(use_models(outputs,prediction_mode))
else:
    for name,match,prev_masteries,masteries,team_nb in matches:
        handled_match = _handle_match(match, prev_masteries, masteries, 0, include_victory=False)

        handled_game_ordered = {}
        for t in handled_match:
            handled_game_ordered[t] = {}
            for role in ["TOP","JUNGLE","MIDLANE","ADC","SUPPORT"]:
                handled_game_ordered[t][role] = handled_match[t][role]
        handled_match = handled_game_ordered

        if live_game:
            with open("last_live_game.json","w") as f:
                json.dump(handled_match,f)
        else:
            with open("last_played_game.json","w") as f:
                json.dump(handled_match,f)

        proper_match1 = dataset_fullgame.json_to_numpy(handled_match)
        # proper_match2 = dataset_big.json_to_numpy(handled_match)
        # proper_match3 = dataset_small.json_to_numpy(handled_match)
        outputs = []
        outputs.append(model0(torch.tensor(proper_match1).unsqueeze(dim=0))[0].item())
        # outputs.append(model1(torch.tensor(proper_match3).unsqueeze(dim=0))[0].item())
        # outputs.append(model2(torch.tensor(proper_match1).unsqueeze(dim=0))[0].item())
        # outputs.append(model3(torch.tensor(proper_match1).unsqueeze(dim=0))[0].item())

        predictions.append(use_models(outputs,prediction_mode,team_nb=team_nb))

print(predictions)