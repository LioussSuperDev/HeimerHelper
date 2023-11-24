from typing import Iterator
import numpy as np
from torch.utils.data import IterableDataset
import json
import os
from os.path import isfile, join
import random
from dataset_utils import rank_to_int,role_to_int
np.set_printoptions(threshold=100000, suppress=True)

player_size = 18
total_size = 10*player_size+1

def json_to_numpy(match):
    #nb games, winrate, champ_nb_games, champ_winrate, champ_nb_games, champ_winrate, 10 games
    returned = np.zeros(total_size)
    nb_ranked = 0
    ranked_points = 0
    pindex = 0
    for team in ["teamA","teamB"]:
        for json_role in ["TOP","JUNGLE","MIDLANE","ADC","SUPPORT"]:
            if not json_role in match[team] or match[team][json_role] == None:
                print("ALERT")
            player = match[team][json_role]

            current_index = pindex*player_size
            if "tier" in player and player["tier"] != None:
                ranked_points += rank_to_int(player["tier"],player["rank"],0)
                nb_ranked += 1

            idx_from_now = current_index
            returned[idx_from_now] = player["wins"]+player["losses"]

            if player["wins"]+player["losses"] > 0:
                returned[idx_from_now+1] = player["wins"]/(player["wins"]+player["losses"])
            else:
                returned[idx_from_now+1] = 0.5

            if "wins" in player["championData"]:
                returned[idx_from_now+2] = player["championData"]["totalMatches"]
                returned[idx_from_now+3] = player["championData"]["wins"]/player["championData"]["totalMatches"]
            else:
                returned[idx_from_now+3] = 0.5
            if "prev_wins" in player["championData"]:
                returned[idx_from_now+4] = player["championData"]["prev_totalMatches"]
                returned[idx_from_now+5] = player["championData"]["prev_wins"]/player["championData"]["prev_totalMatches"]
            else:
                returned[idx_from_now+5] = 0.5

            returned[idx_from_now+6] = player["championData"]["champion_stats_winrate"]
            returned[idx_from_now+7] = player["championData"]["champion_stats_opponent"]
            for i in range(10):
                if i < len(player["matches"]):
                    returned[idx_from_now+8+i] = int(player["matches"][i]["win"])
                else:
                    returned[idx_from_now+8+i] = 0.5

            pindex += 1
    if nb_ranked != 0:
        returned[-1] = ranked_points/nb_ranked
    else:
        returned[-1] = -1000
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