import torch
import torch.nn as nn
import torch.nn.functional as F
class MLP1(nn.Module):
    def __init__(self, input_size):
        super(MLP1, self).__init__()

        self.bn0 = torch.nn.BatchNorm1d(input_size)
        self.fc0 = nn.Linear(input_size, 5000)
        self.relu0 = nn.ReLU()
        self.dropout0 = nn.Dropout(0.4)

        self.bn1 = torch.nn.BatchNorm1d(5000)
        self.fc1 = nn.Linear(5000, 1000)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)

        self.bn2 = torch.nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 500)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.4)

        self.bn3 = torch.nn.BatchNorm1d(500)
        self.fc3 = nn.Linear(500, 100)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.4)

        self.bn4 = torch.nn.BatchNorm1d(100)
        self.fc4 = nn.Linear(100, 50)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.4)

        self.bnfinal = torch.nn.BatchNorm1d(50)
        self.fcfinal = nn.Linear(50, 1)

    def forward(self, x):
        x = x.float()

        x = self.dropout0(self.relu0(self.fc0(self.bn0(x))))
        x = self.dropout1(self.relu1(self.fc1(self.bn1(x))))
        x = self.dropout2(self.relu2(self.fc2(self.bn2(x))))
        x = self.dropout3(self.relu3(self.fc3(self.bn3(x))))
        x = self.dropout4(self.relu4(self.fc4(self.bn4(x))))
        x = F.sigmoid(self.fcfinal(self.bnfinal(x)))

        return x

    @property
    def device(self):
        return next(self.parameters()).device
    

class MLP2(nn.Module):
    def __init__(self, input_size, dropout=0.4):
        super(MLP2, self).__init__()

        self.bn0 = torch.nn.BatchNorm1d(input_size)
        self.fc0 = nn.Linear(input_size, 5000)
        self.relu0 = nn.ReLU()
        self.dropout0 = nn.Dropout(dropout)

        self.bn1 = torch.nn.BatchNorm1d(5000)
        self.fc1 = nn.Linear(5000, 500)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.bn2 = torch.nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 50)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.bnfinal = torch.nn.BatchNorm1d(50)
        self.fcfinal = nn.Linear(50, 1)

    def forward(self, x):
        x = x.float()

        x = self.dropout0(self.relu0(self.fc0(self.bn0(x))))
        x = self.dropout1(self.relu1(self.fc1(self.bn1(x))))
        x = self.dropout2(self.relu2(self.fc2(self.bn2(x))))
        x = F.sigmoid(self.fcfinal(self.bnfinal(x)))

        return x

    @property
    def device(self):
        return next(self.parameters()).device
    
class MLP2_TEAM(nn.Module):
    def __init__(self, input_size):
        super(MLP2_TEAM, self).__init__()

        self.bn0 = torch.nn.BatchNorm1d(input_size)
        self.fc0 = nn.Linear(input_size, 2500)
        self.relu0 = nn.ReLU()

        self.bn1 = torch.nn.BatchNorm1d(2500)
        self.fc1 = nn.Linear(2500, 250)
        self.relu1 = nn.ReLU()

        self.bn2 = torch.nn.BatchNorm1d(250)
        self.fc2 = nn.Linear(250, 50)
        self.relu2 = nn.ReLU()

        self.bnfinal = torch.nn.BatchNorm1d(50)
        self.fcfinal = nn.Linear(50, 1)

    def forward(self, x):
        x = x.float()

        x = (self.relu0(self.fc0(self.bn0(x))))
        x = (self.relu1(self.fc1(self.bn1(x))))
        x = (self.relu2(self.fc2(self.bn2(x))))
        x = F.sigmoid(self.fcfinal(self.bnfinal(x)))

        return x

    @property
    def device(self):
        return next(self.parameters()).device
    
class MLP3(nn.Module):
    def __init__(self, input_size):
        super(MLP3, self).__init__()

        self.bn0 = torch.nn.BatchNorm1d(input_size)
        self.fc0 = nn.Linear(input_size, 10)
        self.relu0 = nn.ReLU()
        self.dropout0 = nn.Dropout(0.4)

        self.bn1 = torch.nn.BatchNorm1d(10)
        self.fc1 = nn.Linear(10, 10)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)

        self.bnfinal = torch.nn.BatchNorm1d(10)
        self.fcfinal = nn.Linear(10, 1)

    def forward(self, x):
        x = x.float()

        x = self.dropout0(self.relu0(self.fc0(self.bn0(x))))
        x = self.dropout1(self.relu1(self.fc1(self.bn1(x))))
        x = F.sigmoid(self.fcfinal(self.bnfinal(x)))

        return x

    @property
    def device(self):
        return next(self.parameters()).device
    

class MLP4(nn.Module):
    def __init__(self, input_size):
        super(MLP4, self).__init__()

        self.bn0 = torch.nn.BatchNorm1d(input_size)
        self.fc0 = nn.Linear(input_size, 1000)
        self.relu0 = nn.ReLU()
        self.dropout0 = nn.Dropout(0.4)

        self.bn1 = torch.nn.BatchNorm1d(1000)
        self.fc1 = nn.Linear(1000, 200)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)

        self.bn2 = torch.nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 50)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.4)

        self.bn3 = torch.nn.BatchNorm1d(50)
        self.fc3 = nn.Linear(50, 10)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.4)

        self.bnfinal = torch.nn.BatchNorm1d(10)
        self.fcfinal = nn.Linear(10, 1)

    def forward(self, x):
        x = x.float()

        x = self.dropout0(self.relu0(self.fc0(self.bn0(x))))
        x = self.dropout1(self.relu1(self.fc1(self.bn1(x))))
        x = self.dropout2(self.relu2(self.fc2(self.bn2(x))))
        x = self.dropout3(self.relu3(self.fc3(self.bn3(x))))
        x = F.sigmoid(self.fcfinal(self.bnfinal(x)))

        return x

    @property
    def device(self):
        return next(self.parameters()).device