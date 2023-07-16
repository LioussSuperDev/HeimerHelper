from typing import Iterator
import numpy as np
import torch
from torch.utils.data import IterableDataset
import json
import os
from os.path import isfile, join
import random

def rank_to_int(tier,rank,lp):

    rank_int = 0
    if rank.lower() == "iii":
        rank_int = 1
    elif rank.lower() == "ii":
        rank_int = 2
    elif rank.lower() == "i":
        rank_int = 3
    tier_int = 0
    if tier.lower() == "bronze":
        tier_int = 4
    elif tier.lower() == "silver":
        tier_int = 8
    elif tier.lower() == "gold":
        tier_int = 12
    elif tier.lower() == "platinum":
        tier_int = 16
    elif tier.lower() == "diamond":
        tier_int = 20
    elif tier.lower() == "master" or tier.lower() == "grandmaster" or tier.lower() == "challenger":
        return 2400 + lp
    return (tier_int + rank_int) * 100 + lp

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
    
player_match_size = 10+5
basic_data_size = 3
role_size = 5
player_size = (basic_data_size+10*player_match_size+role_size)
total_size = 5*player_size

def json_to_numpy(match, team):

    returned = np.zeros(total_size)

    pindex = 0
    for json_role in ["TOP","JUNGLE","MIDLANE","ADC","SUPPORT"]:
        player = match[team][json_role]

        current_index = pindex*player_size

        if "tier" in player:
            returned[current_index] = rank_to_int(player["tier"],player["rank"],player["lp"])
        else:
            returned[current_index] = -1000
        #1

        if len(player["premade"]) > 0:
            returned[current_index+1+role_to_int(player["premade"][0])] = 1
        #6

        # returned[current_index+rank_size+role_size] = player["championData"]["championId"]


        idx_from_now = current_index+1+role_size
        returned[idx_from_now] = player["wins"]+player["losses"]
        #7

        if player["wins"]+player["losses"] > 0:
            returned[idx_from_now+1] = player["wins"]/(player["wins"]+player["losses"])
        else:
            returned[idx_from_now+1] = -1
        #8

        for i,c_match in enumerate(player["matches"]):
            returned[idx_from_now+2+i*player_match_size] = int(c_match["win"])
            returned[idx_from_now+2+i*player_match_size+1] = (c_match["matchDuration"])/3600000
            returned[idx_from_now+2+i*player_match_size+2] = (c_match["kills"])
            returned[idx_from_now+2+i*player_match_size+3] = (c_match["damage"])
            returned[idx_from_now+2+i*player_match_size+4] = (c_match["gold"])
            returned[idx_from_now+2+i*player_match_size+5] = (c_match["deaths"])
            returned[idx_from_now+2+i*player_match_size+6] = (c_match["assists"])
            returned[idx_from_now+2+i*player_match_size+7] = (c_match["cs"])
            returned[idx_from_now+2+i*player_match_size+8] = (c_match["visionScore"])
            returned[idx_from_now+2+i*player_match_size+9] = (c_match["creation_gap"])
            returned[idx_from_now+2+i*player_match_size+10+role_to_int(c_match["role"])] = 1

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
        self.team_side = "teamA"
    def __len__(self):
        return 2*len(self.match_list)

    def __next__(self):
        try:
            file_path = os.path.join(os.path.dirname(__file__), "dataset\\"+self.split+"\\"+self.match_list[self.idx])
            with open(file_path,"r") as f:
                match = json.load(f)
                if self.team_side == "teamA":
                    winner = match.pop("winner")
                else:
                    winner = not match.pop("winner")
                features = json_to_numpy(match,self.team_side)
                self.idx += 1
                return features, np.array(winner).astype(np.float32)
        except:
            if self.team_side == "teamA":
                self.team_side = "teamB"
                self.idx = 0
                return self.__next__()
            else:
                raise StopIteration()
    def __iter__(self) -> Iterator:
        return self