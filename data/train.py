import torch
import torch.nn as nn
import torch.nn.functional as F
import rank_dataset_0

class HHM1(nn.Module):
    def __init__(self, input_size):
        super(HHM1, self).__init__()
        self.bn1 = torch.nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, 20)

        self.bn2 = torch.nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 200)

        self.bn3 = torch.nn.BatchNorm1d(200)
        self.fc3 = nn.Linear(200, 50)

        self.bn4 = torch.nn.BatchNorm1d(50)
        self.fc4 = nn.Linear(50, 10)

        self.bn5 = torch.nn.BatchNorm1d(10)
        self.fc5 = nn.Linear(10, 1)
    def forward(self, x):
        x = F.relu(self.fc1(self.bn1(x)))
        x = F.relu(self.fc2(self.bn2(x)))
        x = F.relu(self.fc3(self.bn3(x)))
        x = F.relu(self.fc4(self.bn4(x)))
        x = F.sigmoid(self.fc5(self.bn5(x)))
        return x
    @property
    def device(self):
        return next(self.parameters()).device
    
def train_one_epoch(training_loader, optimizer, loss_fn, device):
    avg_loss = 0
    for data in training_loader:
        # Every data instance is an input + label pair
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        avg_loss += loss.item()
    return avg_loss/len(training_loader)

dataset = rank_dataset_0.RankDataSet0()

train_size, test_size = int(4*len(dataset)/5), len(dataset) - int(4*len(dataset)/5)
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

training_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model = HHM1(dataset.data_size())
model = model.to(device)
#model.cuda()

epoch_number = 0
EPOCHS = 100

optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
loss_fn = torch.nn.BCELoss()


for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    model.train(True)
    avg_loss = train_one_epoch(training_loader, optimizer, loss_fn, device)
    model.train(False)

    running_vloss = 0.0
    tot = 0
    success = 0
    for i, vdata in enumerate(test_loader):
        vinputs, vlabels = vdata
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)
        voutputs = model(vinputs)

        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss


        tot = i

    avg_vloss = running_vloss / (tot + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    epoch_number += 1