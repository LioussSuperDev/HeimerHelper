from typing import Iterator
import numpy as np
from torch.utils.data import IterableDataset
import json
import os
from os.path import isfile, join
import random
from dataset_utils import rank_to_int,role_to_int

player_match_size = 1
basic_data_size = 7
player_size = (basic_data_size+10*player_match_size)
total_size = 10*player_size

def json_to_numpy(match):

    returned = np.zeros(total_size)

    pindex = 0
    for team in match:
        for json_role in ["TOP","JUNGLE","MIDLANE","ADC","SUPPORT"]:
            player = match[team][json_role]

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
            returned[current_index+5] = player["championData"]["champion_stats_winrate"]
            returned[current_index+6] = player["championData"]["champion_stats_opponent"]
            for i,c_match in enumerate(player["matches"]):
                returned[current_index+7+i*player_match_size] = int(c_match["win"])

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