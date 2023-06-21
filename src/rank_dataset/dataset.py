from typing import Iterator
import numpy as np
import torch
from torch.utils.data import IterableDataset
import json
import os
from os.path import isfile, join
import random

def rank_to_onehot(tier,rank):

    returned = np.zeros(27)
    rank_int = 0
    if rank == "III":
        rank_int = 1
    elif rank == "II":
        rank_int = 2
    elif rank == "I":
        rank_int = 3

    if tier.lower() == "iron":
        returned[rank_int] = 1
    elif tier.lower() == "bronze":
        returned[4+rank_int] = 1
    elif tier.lower() == "silver":
        returned[8+rank_int] = 1
    elif tier.lower() == "gold":
        returned[12+rank_int] = 1
    elif tier.lower() == "platinum":
        returned[16+rank_int] = 1
    elif tier.lower() == "diamond":
        returned[20+rank_int] = 1
    elif tier.lower() == "master":
        returned[24] = 1
    elif tier.lower() == "grandmaster":
        returned[25] = 1
    elif tier.lower() == "challenger":
        returned[26] = 1
    else:
        print("ERROR",tier,rank)
    return returned

def role_to_int(role):
    if role == "TOP":
        return 0
    elif role == "JUNGLE":
        return 1
    elif role == "MIDLANE":
        return 2
    elif role == "ADC":
        return 3
    elif role == "SUPPORT":
        return 4

champion_number = 1000
player_match_size = 11+5
basic_data_size = 3
champion_data_size = champion_number + 9
role_size = 5
rank_size = rank_to_onehot("iron","III").shape[0]
player_size = (rank_size+basic_data_size+10*player_match_size+role_size+champion_data_size)
total_size = 10*player_size

def json_to_numpy(match):

    returned = np.zeros(total_size)

    pindex = 0
    for team in match:
        for player in match[team]:
            player = match[team][player]

            current_index = pindex*player_size
            if "tier" in player:
                returned[current_index:current_index+rank_size] = rank_to_onehot(player["tier"],player["rank"])

            if len(player["premade"]) > 0:
                returned[current_index+rank_size+role_to_int(player["premade"][0])] = 1
            returned[current_index+rank_size+role_size+player["championData"]["championId"]] = 1

            idx_from_now = current_index+rank_size+role_size+champion_number
            returned[idx_from_now] = player["wins"]
            returned[idx_from_now+1] = player["losses"]
            if "lp" in player:
                returned[idx_from_now+2] = player["lp"]
            if "wins" in player["championData"]:
                returned[idx_from_now+3] = player["championData"]["wins"]
                returned[idx_from_now+4] = player["championData"]["totalMatches"]
                returned[idx_from_now+5] = player["championData"]["lpAvg"]
                returned[idx_from_now+6] = player["championData"]["csPerMatch"]
                returned[idx_from_now+7] = player["championData"]["damagePerMatch"]
                returned[idx_from_now+8] = player["championData"]["deathsPerMatch"]
                returned[idx_from_now+9] = player["championData"]["killsPerMatch"]
                returned[idx_from_now+10] = player["championData"]["assistsPerMatch"]
                returned[idx_from_now+11] = player["championData"]["goldPerMatch"]

            for i,c_match in enumerate(player["matches"]):
                returned[idx_from_now+12+i*player_match_size] = int(c_match["win"])
                returned[idx_from_now+12+i*player_match_size+1] = (c_match["matchDuration"])
                returned[idx_from_now+12+i*player_match_size+2] = (c_match["championId"])
                returned[idx_from_now+12+i*player_match_size+3] = (c_match["kills"])
                returned[idx_from_now+12+i*player_match_size+4] = (c_match["damage"])
                returned[idx_from_now+12+i*player_match_size+5] = (c_match["gold"])
                returned[idx_from_now+12+i*player_match_size+6] = (c_match["deaths"])
                returned[idx_from_now+12+i*player_match_size+7] = (c_match["assists"])
                returned[idx_from_now+12+i*player_match_size+8] = (c_match["cs"])
                returned[idx_from_now+12+i*player_match_size+9] = (c_match["visionScore"])
                returned[idx_from_now+12+i*player_match_size+10] = (c_match["creation_gap"])
                returned[idx_from_now+12+i*player_match_size+11+role_to_int(c_match["role"])] = 1
            pindex += 1

    return returned

def get_datasize():
    return total_size

class RankDataSet0(IterableDataset):

    def __init__(self, split="train", transform=None):
        super(RankDataSet0).__init__()
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