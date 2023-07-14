from typing import Iterator
import numpy as np
from torch.utils.data import IterableDataset
import json
import os
from os.path import isfile, join
import random

def rank_to_int(tier,rank,lp):

    rank_int = 0
    if rank == "III":
        rank_int = 1
    elif rank == "II":
        rank_int = 2
    elif rank == "I":
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

player_match_size = 1
basic_data_size = 5
player_size = (basic_data_size+10*player_match_size)
total_size = 10*player_size

def json_to_numpy(match):

    returned = np.zeros(total_size)

    pindex = 0
    for team in match:
        for player in match[team]:
            player = match[team][player]

            current_index = pindex*player_size

            #1
            if "tier" in player:
                returned[current_index] = rank_to_int(player["tier"],player["rank"],player["lp"])
            else:
                returned[current_index] = -1000

            #2
            returned[current_index+1] = player["wins"]+player["losses"]

            #3
            if player["wins"]+player["losses"] > 0:
                returned[current_index+2] = player["wins"]/(player["wins"]+player["losses"])
            else:
                returned[current_index+2] = -1

            if "wins" in player["championData"]:
                #4
                returned[current_index+3] = player["championData"]["totalMatches"]

                #5
                returned[current_index+4] = player["championData"]["wins"]/player["championData"]["totalMatches"]

            for i,c_match in enumerate(player["matches"]):
                returned[current_index+5+i*player_match_size] = int(c_match["win"])

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