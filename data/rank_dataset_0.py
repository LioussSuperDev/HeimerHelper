import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os
from os.path import isfile, join

class RankDataSet0(Dataset):

    def __init__(self, transform=None):
        self.match_list = [f for f in os.listdir("rank_dataset_0") if isfile(join("rank_dataset_0", f))]
        self.transform = transform

    def __len__(self):
        return len(self.match_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = os.path.join(os.path.dirname(__file__), "rank_dataset_0\\"+self.match_list[idx])
        with open(file_path,"r") as f:
            match = json.load(f)
            winner = match.pop("winner")

            for i in range(10):
                match.pop("player_"+str(i)+"_champion")

            return np.array(list(match.values())).astype(np.float32), np.array([2-winner]).astype(np.float32)

    def data_size(self):
        file_path = os.path.join(os.path.dirname(__file__), "rank_dataset_0\\"+self.match_list[0])
        with open(file_path,"r") as f:
            match = json.load(f)

            for i in range(10):
                match.pop("player_"+str(i)+"_champion")

            return len(match.values())-1