import torch
import torch.nn as nn
import torch.nn.functional as F
import dataset
import os
import model_architectures
import torch_directml
import warnings

warnings.filterwarnings("ignore")

dml = torch_directml.device()

BATCH_SIZE = 512

def train_one_epoch(training_loader, optimizer, loss_fn, device):
    nb_batches = len(training_loader)
    avg_loss = 0
    success = 0
    tot = 0
    for idx,data in enumerate(training_loader):
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        given_outputs = outputs.squeeze()

        loss = loss_fn(given_outputs, labels)
        loss.backward()

        optimizer.step()

        avg_loss += loss.item()

        rounded = torch.round(outputs)
        tot += labels.shape[0]

        l_succes = 0

        for i in range(rounded.shape[0]):
            if rounded[i,0] == labels[i]:
                success += 1
                l_succes += 1

        print(str(round((idx+1)*100/nb_batches,2)),"% | loss/acc :",loss.item(),round(l_succes/rounded.shape[0],2),"                                                 ",end="\r")
    return avg_loss/len(training_loader),success/tot



#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
device = torch_directml.device()

max_accuracy = 0

train_dataset = dataset.RankDataSet0(split="train")
test_dataset = dataset.RankDataSet0(split="test")

print("Starting... data size :",dataset.get_datasize())
print("Number of training exemples :",len(train_dataset))
learning_rates = [0.01,0.005,0.001,0.0007,0.0005,0.0001,0.00005]
weight_decays = [0,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05]

for wd in weight_decays:
    for lr in learning_rates:
        print("======= LR :",lr,":: WD",wd,"=======")
        model = model_architectures.MLP1(dataset.get_datasize())
        model = model.to(device)

        epoch_number = 0
        EPOCHS = 30

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        loss_fn = torch.nn.BCELoss()

        for epoch in range(EPOCHS):
            train_dataset = dataset.RankDataSet0(split="train")
            test_dataset = dataset.RankDataSet0(split="test")
            training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

            print("epoch",str(epoch)+"/"+str(EPOCHS))
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
                torch.save(model.state_dict(), "models/"+str(round(acc,4))+"_l"+str(lr)+"_w"+str(wd)+"_bsize"+str(BATCH_SIZE)+".state")