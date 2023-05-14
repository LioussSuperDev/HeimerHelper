import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import dataset
import sys
import dataset_utils

sys.path.insert(0, '..')
import gatherer_utils
sys.path.remove('..')
dataset = dataset.RankDataSet0()

model = models.HHM2(dataset.data_size())
model.load_state_dict(torch.load("models\\0.7980.state"))
model.eval()

sys.path.insert(0, '..')
from utils import api_getter
watcher = api_getter.get_watcher()

def check_game(region, seed_player_name):
    sumDTO = watcher.summoner.by_name(region,seed_player_name)
    spec_game = watcher.spectator.by_summoner(sumDTO.id)

    for p in spec_game["participants"]:
        psumDTO = watcher.summoner.by_id(region, p["summonerId"])
        player_data = gatherer_utils.load_player_data(region, psumDTO)
        p_name = player_data["player"]["name"]

        ok = False
        done = []
        while not ok:
            lane = input("Lane of "+p_name+" (top/jng/mid/adc/sup) : ")
            t_lane = None
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
            if t_lane != None:
                if not t_lane+str(p["teamId"]) in done:
                    done.append(t_lane+str(p["teamId"]))
                    player_cleared_data = dataset_utils.handle_player(player_data,lane,p["championId"],p["teamId"],player_data["player"]["summonerLevel"])
                    ok = True
                else:
                    print("This lane is already set for this team.")
            else:
                print("Unknown lane.")

#12 requests max


check_game("euw1",["TDS IReaz"])