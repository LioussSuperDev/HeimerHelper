import torch
import torch.nn as nn
import torch.nn.functional as F
import dataset
import os
import model_architectures

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

dataset = dataset.RankDataSet0()

train_size, test_size = int(4*len(dataset)/5), len(dataset) - int(4*len(dataset)/5)
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

training_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
max_accuracy = 0
for test_index in range(50000):
    model = model_architectures.HHM2(dataset.data_size())
    model = model.to(device)
    if torch.cuda.is_available():
        model.cuda()

    epoch_number = 0
    EPOCHS = 40

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0) #FOR HHM1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0075, weight_decay=0) #FOR HHM2, batch_size=4

    loss_fn = torch.nn.BCELoss()

    
    for epoch in range(EPOCHS):
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
        print('{} LOSS train {} valid {}'.format(test_index, round(avg_loss,3), round(avg_vloss,3)))
        #print('ACC train {} valid {}'.format(train_acc, acc))
        print('{} Validation accuracy/max : {}/{}'.format(test_index, round(acc,3),round(max_accuracy,3)))
        epoch_number += 1
        os.makedirs("models", exist_ok=True)
        if acc == max_accuracy:
            torch.save(model.state_dict(), "models/"+str(round(acc,4))+".state")