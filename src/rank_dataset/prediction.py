import torch
import torch.nn as nn
import torch.nn.functional as F
import model_architectures
import dataset
import sys
import dataset_utils
import numpy as np

sys.path.insert(0, '..')
import gatherer_utils
sys.path.remove('..')
dataset = dataset.RankDataSet0()

model = model_architectures.HHM2(dataset.data_size())
model.load_state_dict(torch.load("models\\0.7980.state", map_location=torch.device('cpu')))
model.eval()

sys.path.insert(0, '..')
from utils import api_getter
watcher = api_getter.get_watcher()

def check_games(region, player_name=None, match_id=None, queue_type="RANKED_SOLO_5x5", auto=False, mute_auto_role=False): #"RANKED_FLEX_SR" or "RANKED_SOLO_5X5"
    print("Predicting for "+(match_id or player_name))
    sumDTO = watcher.summoner.by_name(region,player_name)
    spec_game = None
    if match_id != None:
        spec_game = watcher.match.by_id(region, match_id)
    elif player_name != None:
        spec_game = watcher.spectator.by_summoner(region, sumDTO["id"])
    else:
        print("Please choose a player name or a match id")
        return
    total = {}
    done = []

    team_1_content = []
    team_2_content = []

    array = None
    if match_id != None:
        array = spec_game["info"]["participants"]
    elif player_name != None:
        array = spec_game["participants"]

    for p_index,p in enumerate(array):
        pRSO = watcher.summoner.by_id(region, p["summonerId"])
        player_data = gatherer_utils.load_player_data(region, pRSO["puuid"])
        p_name = player_data["player"]["name"]

        ok = False
        
        while not ok:
            t_lane = None  
            if not auto:
                lane = input("Lane of ["+str(int(p["teamId"]/100))+"]"+p_name+" (top/jng/mid/adc/sup) : ")
                match lane:
                    case "top":
                        t_lane = "TOP"
                    case "mid":
                        t_lane = "MIDDLE"
                    case "jng":
                        t_lane = "JUNGLE"
                    case "adc":
                        t_lane = "BOTTOM"
                    case "sup":
                        t_lane = "UTILITY"
            else:
                if p_index%5 == 0:
                    t_lane = "TOP"
                elif p_index%5 == 1:
                    t_lane = "JUNGLE"
                elif p_index%5 == 2:
                    t_lane = "MIDDLE"
                elif p_index%5 == 3:
                    t_lane = "BOTTOM"
                elif p_index%5 == 4:
                    t_lane = "UTILITY"
                if not mute_auto_role:
                    print("Auto-detecting ["+str(int(p["teamId"]/100))+"]"+p_name+" as "+t_lane)

            if t_lane != None:
                if not (t_lane+str(p["teamId"]) in done):
                    done.append(t_lane+str(p["teamId"]))
                    player_cleared_data = dataset_utils.handle_player(player_data,t_lane,p["championId"],p["teamId"],player_data["player"]["summonerLevel"],queue_type)

                    if p["teamId"] == 100:
                        team_1_content.append(p["summonerName"])
                    else:
                        team_2_content.append(p["summonerName"])
                    for key in player_cleared_data:
                        total[key] = player_cleared_data[key]

                    ok = True
                else:
                    print("This lane is already set for this team.")
            else:
                print("Unknown lane.")
    data = np.array([list(total.values())]).astype(np.float32)
    result = model(torch.tensor(data))
    result_float = result.item()

    t1_win = "won"
    t2_win = "lost"
    if result_float <= 0.5:
        t1_win = "lost"
        t2_win = "won"
        result_float = 1 - result_float
    print()
    print("Team 1 ("+t1_win+") : "+str(team_1_content))
    print("Team 2 ("+t2_win+") : "+str(team_2_content))
    print("Likelihood : "+str(result_float))
    print()

#check_games("euw1",player_name="Le bon jus", auto=False, mute_auto_role=False)
check_games("euw1",match_id="EUW1_6405016678", auto=True, mute_auto_role=False)