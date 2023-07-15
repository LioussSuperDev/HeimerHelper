import torch
import numpy as np
import model_architectures
import dataset_medium
import torch_directml

def test_model(model, test_loader, device):

    cert_accuracies = [[0,0,0,x/100] for x in range(50,105,5)]

    for vdata in test_loader:
        vinputs, vlabels = vdata
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)

        voutputs = model(vinputs)

        rounded = torch.round(voutputs)

        for j in range(vlabels.shape[0]):
            certitude = ((abs(0.5-voutputs[j].item()))+0.5)

            for cert in cert_accuracies:
                if  cert[3]-0.05 <= certitude <= cert[3]+0.05:
                    if rounded[j,0].item() == vlabels[j].item():
                        cert[1] += 1
                    cert[2] += 1
    for cert in cert_accuracies:
        if cert[2] != 0:
            cert[0] = round(100*cert[1]/cert[2],2)
        cert[3] = int(cert[3]*100)
    return cert_accuracies


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
device = torch_directml.device()

model = model_architectures.MLP2(dataset_medium.get_datasize())
model.load_state_dict(torch.load("models\\MLP2\\0.8211_l0.005_w0.001_dsetmedium.state", map_location=device))
model.eval()

test_dataset = dataset_medium.RankDataSet(split="test")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512)
cert_accuracies = test_model(model, test_loader, device)
print("ACCURACY | CERTITUDE")
for cert in cert_accuracies:
    print(cert[0],cert[3])