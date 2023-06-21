import torch
import torch.nn as nn
import torch.nn.functional as F
import dataset
import os
import model_architectures

BATCH_SIZE = 16

def train_one_epoch(training_loader, optimizer, loss_fn, device):
    avg_loss = 0
    success = 0
    tot = 0
    for i,data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs.squeeze(), labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        avg_loss += loss.item()

        

        rounded = torch.round(outputs)
        tot += labels.shape[0]

        l_succes = 0

        for i in range(rounded.shape[0]):
            if rounded[i,0] == labels[i]:
                success += 1
                l_succes += 1

        print("loss/acc :",loss.item(),round(l_succes/rounded.shape[0],2),"                                                 ",end="\r")
    return avg_loss/len(training_loader),success/tot



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
max_accuracy = 0
print("Starting... data size :",dataset.get_datasize())

learning_rates = [0.01,0.005,0.001,0.0005,0.0001]
weight_decays = [0,0.1,0.01,0.001,0.0001,0.00001]

for lr in learning_rates:
    for wd in weight_decays:
        model = model_architectures.HHM2(dataset.get_datasize())
        model = model.to(device)

        if torch.cuda.is_available():
            model.cuda()

        epoch_number = 0
        EPOCHS = 40

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        loss_fn = torch.nn.BCELoss()

        
        for epoch in range(EPOCHS):
            train_dataset = dataset.RankDataSet0(split="train")
            test_dataset = dataset.RankDataSet0(split="test")
            training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

            print("epoch",epoch)
            model.train(True)
            avg_loss,train_acc = train_one_epoch(training_loader, optimizer, loss_fn, device)
            print()
            model.train(False)

            running_vloss = 0.0
            success = 0
            tot_acc = 0
            for i, vdata in enumerate(test_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)

                voutputs = model(vinputs)

                vloss = loss_fn(voutputs, vlabels.unsqueeze(dim=1))
                running_vloss += vloss.item()

                rounded = torch.round(voutputs)
                for j in range(vlabels.shape[0]):
                    if rounded[j,0].item() == vlabels[j].item():
                        success += 1
                    tot_acc+=1


            avg_vloss = running_vloss / (i+1)
            acc = success / tot_acc
            max_accuracy = max(max_accuracy,acc)
            print('lr {} wd {} LOSS train {} valid {}'.format(lr,wd, round(avg_loss,3), round(avg_vloss,3)))
            #print('ACC train {} valid {}'.format(train_acc, acc))
            print('lr {} wd {} Validation accuracy/max : {}/{}'.format(lr,wd, round(acc,3),round(max_accuracy,3)))
            epoch_number += 1
            os.makedirs("models", exist_ok=True)
            if acc == max_accuracy:
                torch.save(model.state_dict(), "models/"+str(round(acc,4))+"_l"+str(lr)+"_w"+str(wd)+".state")