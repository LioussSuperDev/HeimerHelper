import torch
import torch.nn as nn
import torch.nn.functional as F
import rank_dataset_0

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
        self.fc0 = nn.Linear(input_size, 20)

        self.bn1 = torch.nn.BatchNorm1d(20)
        self.fc1 = nn.Linear(20, 20)

        self.bn2 = torch.nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 20)

        self.bn3 = torch.nn.BatchNorm1d(20)
        self.fc3 = nn.Linear(20, 20)

        self.bn4 = torch.nn.BatchNorm1d(20)
        self.fc4 = nn.Linear(20, 20)

        self.bn5 = torch.nn.BatchNorm1d(20)
        self.fc5 = nn.Linear(20, 20)

        self.bn6 = torch.nn.BatchNorm1d(20)
        self.fc6 = nn.Linear(20, 20)

        self.bn7 = torch.nn.BatchNorm1d(20)
        self.fc7 = nn.Linear(20, 20)

        self.bn8 = torch.nn.BatchNorm1d(20)
        self.fc8 = nn.Linear(20, 20)

        self.bnfinal = torch.nn.BatchNorm1d(20)
        self.fcfinal = nn.Linear(20, 1)
    def forward(self, x):
        x = F.relu(self.fc0(self.bn0(x)))
        #x = F.relu(self.fc1(self.bn1(x)))
        #x = F.relu(self.fc2(self.bn2(x)))
        #x = F.relu(self.fc3(self.bn3(x)))
        #x = F.relu(self.fc4(self.bn4(x)))
        #x = F.relu(self.fc5(self.bn5(x)))
        x = F.relu(self.fc6(self.bn6(x)))
        x = F.relu(self.fc7(self.bn7(x)))
        x = F.relu(self.fc8(self.bn8(x)))
        x = F.sigmoid(self.fcfinal(self.bnfinal(x)))
        return x
    @property
    def device(self):
        return next(self.parameters()).device

def train_one_epoch(training_loader, optimizer, loss_fn, device):
    avg_loss = 0
    success = 0
    tot = 0
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
        rounded = torch.round(outputs)
        tot += labels.shape[0]
        for i in range(rounded.shape[0]):
            if rounded[i,0] == labels[i,0]:
                success += 1
    return avg_loss/len(training_loader),success/tot

dataset = rank_dataset_0.RankDataSet0()

train_size, test_size = int(4*len(dataset)/5), len(dataset) - int(4*len(dataset)/5)
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

training_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

model = HHM1(dataset.data_size())
model = model.to(device)
model.cuda()

epoch_number = 0
EPOCHS = 500

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001) #FOR HHM1
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001) #FOR HHM2, batch_size=4

loss_fn = torch.nn.BCELoss()

max_accuracy = 0
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    model.train(True)
    avg_loss,train_acc = train_one_epoch(training_loader, optimizer, loss_fn, device)
    model.train(False)

    running_vloss = 0.0
    success = 0
    tot_acc = 0
    for i, vdata in enumerate(test_loader):
        vinputs, vlabels = vdata
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)

        voutputs = model(vinputs)

        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss.item()

        rounded = torch.round(voutputs)
        for i in range(vlabels.shape[0]):
            if rounded[i,0].item() == vlabels[i,0].item():
                success += 1
            tot_acc+=1


    avg_vloss = running_vloss / len(test_loader)
    acc = success / tot_acc
    max_accuracy = max(max_accuracy,acc)
    print('LOSS train {} valid {}'.format(round(avg_loss,3), round(avg_vloss,3)))
    #print('ACC train {} valid {}'.format(train_acc, acc))
    print('Validation accuracy/max : {}/{}'.format(round(acc,2),round(max_accuracy,2)))
    epoch_number += 1