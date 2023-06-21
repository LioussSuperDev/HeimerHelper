import torch
import torch.nn as nn
import torch.nn.functional as F

class HHM1(nn.Module):
    def __init__(self, input_size):
        super(HHM1, self).__init__()
        self.bn_start = torch.nn.BatchNorm1d(input_size)
        self.cnn1 = torch.nn.Conv1d(1, 16, 6, 6)

        self.bn1 = torch.nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(178, 150)

        self.bn2 = torch.nn.BatchNorm1d(150)
        self.fc2 = nn.Linear(150, 150)

        self.bn3 = torch.nn.BatchNorm1d(150)
        self.fc3 = nn.Linear(150, 50)

        self.bnfinal = torch.nn.BatchNorm1d(50)
        self.fcfinal = nn.Linear(50, 1)
    def forward(self, x):
        x_start = x
        x = self.cnn1(x.unsqueeze(1))
        x = x.flatten(1,2)
        x = F.relu(self.fc1(torch.cat((self.bn1(x),self.bn_start(x_start)),1)))
        x = F.relu(self.fc2(self.bn2(x)))
        x = F.relu(self.fc3(self.bn3(x)))
        x = F.sigmoid(self.fcfinal(self.bnfinal(x)))
        return x
    @property
    def device(self):
        return next(self.parameters()).device


class HHM2(nn.Module):
    def __init__(self, input_size):
        super(HHM2, self).__init__()

        self.bn0 = torch.nn.BatchNorm1d(input_size)
        self.fc0 = nn.Linear(input_size, 5000)

        self.bn1 = torch.nn.BatchNorm1d(5000)
        self.fc1 = nn.Linear(5000, 1000)

        self.bn2 = torch.nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 500)

        self.bn3 = torch.nn.BatchNorm1d(500)
        self.fc3 = nn.Linear(500, 100)

        self.bn4 = torch.nn.BatchNorm1d(100)
        self.fc4 = nn.Linear(100, 50)

        self.bnfinal = torch.nn.BatchNorm1d(50)
        self.fcfinal = nn.Linear(50, 1)
    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc0(self.bn0(x)))
        x = F.relu(self.fc1(self.bn1(x)))
        x = F.relu(self.fc2(self.bn2(x)))
        x = F.relu(self.fc3(self.bn3(x)))
        x = F.relu(self.fc4(self.bn4(x)))
        x = F.sigmoid(self.fcfinal(self.bnfinal(x)))
        return x
    @property
    def device(self):
        return next(self.parameters()).device