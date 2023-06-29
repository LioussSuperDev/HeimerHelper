import torch
import torch.nn as nn
import torch.nn.functional as F
import dataset
import smalldataset
import os
import model_architectures
#import torch_directml
import warnings

warnings.filterwarnings("ignore")


BATCH_SIZE = 512
EPOCHS = 50

def train_one_epoch(training_loader, optimizer, loss_fn, device):
    nb_batches = len(training_loader)
    avg_loss = 0
    for idx,data in enumerate(training_loader):
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        given_outputs = outputs.squeeze()

        loss = loss_fn(given_outputs, inputs)
        loss.backward()

        optimizer.step()

        avg_loss += loss.item()

        rounded = torch.round(outputs)

        print(str(round((idx+1)*100/nb_batches,2)),"% | loss/acc :",round(loss.item(),3),"                                                 ",end="\r")
    return avg_loss/len(training_loader)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
#device = torch_directml.device()

train_dataset = dataset.RankDataSet(split="train")
test_dataset = dataset.RankDataSet(split="test")

print("Starting... data size :",dataset.get_datasize())
print("Number of training exemples :",len(train_dataset))

learning_rates = [0.01]
weight_decays = [0]
models = [(model_architectures.MLP1_Autoencoder,"MLP1_Autoencoder")]
dsets = [(dataset,"big")]

for model_type,model_name in models:

    max_vloss = 0

    for dset,dset_name in dsets:

        dsetsize = dset.get_datasize()

        for wd in weight_decays:
            for lr in learning_rates:
                print("======= MODEL :",model_name,"DATASET",dset_name," LR :",lr,":: WD",wd,"=======")
                print()
                model = model_type(dsetsize)
                model = model.to(device)

                epoch_number = 0
                

                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

                loss_fn = torch.nn.MSELoss()

                for epoch in range(EPOCHS):
                    train_dataset = dset.RankDataSet(split="train")
                    test_dataset = dset.RankDataSet(split="test")
                    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
                    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

                    print("-- EPOCH",str(epoch+1)+"/"+str(EPOCHS)+" --")
                    model.train(True)
                    avg_tloss,train_acc = train_one_epoch(training_loader, optimizer, loss_fn, device)
                    model.train(False)
                    running_vloss = 0.0

                    for i, vdata in enumerate(test_loader):
                        vinputs, vlabels = vdata
                        vinputs = vinputs.to(device)
                        vlabels = vlabels.to(device)

                        voutputs = model(vinputs)

                        vloss = loss_fn(voutputs, vlabels.unsqueeze(dim=1))
                        running_vloss += vloss.item()


                    avg_vloss = running_vloss / (i+1)
                    max_vloss = max(avg_vloss,max_vloss)
                    print('TRAIN :      {}                                                              '.format(round(avg_tloss,2)))
                    print('VALIDATION : {}'.format(round(avg_vloss,3)))
                    print()
                    epoch_number += 1
                    os.makedirs("models/"+model_name, exist_ok=True)
                    if avg_vloss == max_vloss:
                        torch.save(model.state_dict(), "models/"+model_name+"/"+str(round(avg_vloss,4))+"_l"+str(lr)+"_w"+str(wd)+"_dset"+dset_name+".state")