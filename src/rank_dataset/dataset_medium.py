from typing import Iterator
import numpy as np
import torch
from torch.utils.data import IterableDataset
import json
import os
from os.path import isfile, join
import random
from dataset_utils import rank_to_int,role_to_int

    
player_match_size = 10+5
basic_data_size = 4
champion_data_size = 8
role_size = 5
player_size = (basic_data_size+10*player_match_size+role_size+champion_data_size)
total_size = 10*player_size

def json_to_numpy(match):

    returned = np.zeros(total_size)

    pindex = 0
    for team in match:
        for json_role in ["TOP","JUNGLE","MIDLANE","ADC","SUPPORT"]:
            player = match[team][json_role]

            current_index = pindex*player_size
            if "tier" in player:
                returned[current_index] = rank_to_int(player["tier"],player["rank"],player["lp"])
            else:
                returned[current_index] = -1000

            if len(player["premade"]) > 0:
                returned[current_index+1+role_to_int(player["premade"][0])] = 1


            # returned[current_index+rank_size+role_size] = player["championData"]["championId"]


            idx_from_now = current_index+1+role_size
            returned[idx_from_now] = player["wins"]+player["losses"]

            if player["wins"]+player["losses"] > 0:
                returned[idx_from_now+1] = player["wins"]/(player["wins"]+player["losses"])
            else:
                returned[idx_from_now+1] = -1

            if "wins" in player["championData"]:
                returned[idx_from_now+3] = player["championData"]["totalMatches"]
                returned[idx_from_now+4] = player["championData"]["wins"]/player["championData"]["totalMatches"]
                returned[idx_from_now+5] = player["championData"]["csPerMatch"]
                returned[idx_from_now+6] = player["championData"]["damagePerMatch"]
                returned[idx_from_now+7] = player["championData"]["deathsPerMatch"]
                returned[idx_from_now+8] = player["championData"]["killsPerMatch"]
                returned[idx_from_now+9] = player["championData"]["assistsPerMatch"]
                returned[idx_from_now+10] = player["championData"]["goldPerMatch"]

            for i,c_match in enumerate(player["matches"]):
                returned[idx_from_now+11+i*player_match_size] = int(c_match["win"])
                returned[idx_from_now+11+i*player_match_size+1] = (c_match["matchDuration"])/3600000
                returned[idx_from_now+11+i*player_match_size+2] = (c_match["kills"])
                returned[idx_from_now+11+i*player_match_size+3] = (c_match["damage"])
                returned[idx_from_now+11+i*player_match_size+4] = (c_match["gold"])
                returned[idx_from_now+11+i*player_match_size+5] = (c_match["deaths"])
                returned[idx_from_now+11+i*player_match_size+6] = (c_match["assists"])
                returned[idx_from_now+11+i*player_match_size+7] = (c_match["cs"])
                returned[idx_from_now+11+i*player_match_size+8] = (c_match["visionScore"])
                returned[idx_from_now+11+i*player_match_size+9] = (c_match["creation_gap"])
                returned[idx_from_now+11+i*player_match_size+10+role_to_int(c_match["role"])] = 1

            pindex += 1
    return returned

def get_datasize():
    return total_size

class RankDataSet(IterableDataset):

    def __init__(self, split="train", transform=None):
        super(RankDataSet).__init__()
        self.match_list = ([f for f in os.listdir(os.path.join(os.path.dirname(__file__), "dataset\\"+split)) if isfile(join(os.path.join(os.path.dirname(__file__), "dataset\\"+split), f))])
        random.shuffle(self.match_list)
        self.transform = transform
        self.split = split
        self.idx = 0

    def __len__(self):
        return len(self.match_list)

    def __next__(self):
        try:
            file_path = os.path.join(os.path.dirname(__file__), "dataset\\"+self.split+"\\"+self.match_list[self.idx])
            with open(file_path,"r") as f:
                match = json.load(f)
                winner = match.pop("winner")
                features = json_to_numpy(match)
                self.idx += 1
                return features, np.array(winner).astype(np.float32)
        except:
            raise StopIteration()
    def __iter__(self) -> Iterator:
        return self