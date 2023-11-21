import torch
import dataset_fullgame
import dataset_fullgame_limited
import dataset_teamonly
import dataset_teamonly_champions
import os
import model_architectures
# import torch_directml
import warnings

warnings.filterwarnings("ignore")


BATCH_SIZE = 256
EPOCHS = 50

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

        print(str(round((idx+1)*100/nb_batches,2)),"% | loss/acc :",round(loss.item(),3),round(l_succes/rounded.shape[0],2),"                                                 ",end="\r")
    return avg_loss/len(training_loader),success/tot



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
# device = torch_directml.device()
dset,dset_name = dataset_fullgame_limited,"dataset_fullgame_limited"
# dset,dset_name = dataset_teamonly,"dataset_teamonly"
train_dataset = dset.RankDataSet(split="train")

print("Starting... data size :",dset.get_datasize())
print("Number of training exemples :",len(train_dataset))

learning_rates = [0.002,0.001,0.004,0.006,0.01,0.02]
weight_decays = [0.002,0.0001,0.01]
models = [(model_architectures.MLP2,"MLP2"),(model_architectures.MLP3,"MLP3")]
dsets = [(dset,dset_name)]

for model_type,model_name in models:

    max_accuracy = 0

    for dset,dset_name in dsets:

        dsetsize = dset.get_datasize()

        for wd in weight_decays:
            for lr in learning_rates:
                print("======= MODEL :",model_name,"DATASET",dset_name," LR :",lr,":: WD",wd,"=======")
                print()
                model = model_type(dsetsize)
                model = model.to(device)
    
                epoch_number = 0
                

                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

                loss_fn = torch.nn.BCELoss()

                for epoch in range(EPOCHS):
                    train_dataset = dset.RankDataSet(split="train")
                    test_dataset = dset.RankDataSet(split="test")
                    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
                    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

                    #Training
                    print("-- EPOCH",str(epoch+1)+"/"+str(EPOCHS)+" --")
                    model.train(True)
                    avg_tloss,train_acc = train_one_epoch(training_loader, optimizer, loss_fn, device)
                    
                    
                    #Testing
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

                    
                    print('TRAIN :      {} - {}%                                                              '.format(round(avg_tloss,3), round(train_acc*100,2)))
                    print('VALIDATION : {} - {}%/{}%'.format(round(avg_vloss,3),round(acc*100,2),round(max_accuracy*100,2)))
                    print()
                    epoch_number += 1
                    os.makedirs("models/"+dset_name+"/"+model_name, exist_ok=True)
                    if acc == max_accuracy:
                        torch.save(model.state_dict(), "models/"+dset_name+"/"+model_name+"/"+str(round(acc,4))+"_l"+str(lr)+"_w"+str(wd)+"_dset"+dset_name+".state")
