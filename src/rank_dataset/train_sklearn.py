import torch
import dataset_fullgame
import dataset_teamonly
import dataset_teamonly_champions
import dataset_fullgame_limited
from sklearn.linear_model import LogisticRegression
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import os


models = [(xgb.XGBClassifier(learning_rate=i, objective="binary:logistic"),"XGBRegressor lr"+str(i)) for i in [0.1,0.2,0.3,0.01,0.001,0.5,0.05,0.005]]
models += [(GradientBoostingClassifier(learning_rate=0.1,n_estimators=i),"GradBoosting n_estimators"+str(i)) for i in [25,100,500,1000]]
models += [(LogisticRegression(penalty="l2",C=i,max_iter=10000, random_state=42),"LOGISTIC default-"+str(i)) for i in [0.1,0.01,0.001]]
models += [(LogisticRegression(solver="liblinear",penalty="l1",C=i,max_iter=10000, random_state=42),"LOGISTIC liblinearL1-"+str(i)) for i in [0.1,0.01,0.001]]
models += [(LogisticRegression(solver="saga",penalty="l1",C=i,max_iter=10000, random_state=42),"LOGISTIC saga-"+str(i)) for i in [0.1,0.01,0.001]]
models += [(RandomForestClassifier(criterion='gini', max_depth=i),"RandomForest "+str(i)) for i in [None,4,5,6,7,8,9,10,11,12,13,14,15]]

############################### DATASET

dset = dataset_fullgame_limited
dset_name = "dataset_fullgame_limited"

# dset = dataset_fullgame
# dset_name = "dataset_fullgame"

# dset = dataset_teamonly
# dset_name = "dataset_teamonly"

# dset = dataset_teamonly_champions
# dset_name = "dataset_team_only_champions"

#######################################

train_dataset = dset.RankDataSet(split="train")
test_dataset = dset.RankDataSet(split="test")

training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)

print("Starting... data size :",dset.get_datasize())
print("Number of training exemples :",len(train_dataset))

train_X = []
train_Y = []
test_X = []
test_Y = []

#Reading train dataset
for i, vdata in enumerate(training_loader):
    vinputs, vlabels = np.array(vdata[0]),np.array(vdata[1])
    train_X.append(vinputs)
    train_Y.append(vlabels)

train_X = np.vstack(train_X)
train_Y = np.hstack(train_Y)

scaler = StandardScaler()
scaler.fit(train_X)
save_path = "models/"+dset_name+"/"
os.makedirs(save_path,exist_ok=True)
pickle.dump(scaler,open(save_path+"/scaler.pickle","wb"))
train_X = scaler.transform(train_X)

print("Train dataset loaded")

#Reading train dataset
for i, vdata in enumerate(test_loader):
    vinputs, vlabels = np.array(vdata[0]),np.array(vdata[1])
    test_X.append(vinputs)
    test_Y.append(vlabels)

test_X = np.vstack(test_X)
test_Y = np.hstack(test_Y)
test_X = scaler.transform(test_X)

print("Test dataset loaded")
print()

save_path = save_path+"/sklearn"
os.makedirs(save_path,exist_ok=True)


for model_type,model_name in models:

    print("======= MODEL :",model_name,"=======")
    print("fitting...")

    model_type = model_type.fit(train_X,train_Y)

    print("predicting...")
    test_pred = model_type.predict(test_X)

    #computing accuracy
    accuracy = 0
    for i in range(test_pred.shape[0]):
        if test_pred[i] == test_Y[i]:
            accuracy += 1
    accuracy /= test_pred.shape[0]
    print("accuracy",round(accuracy,3))
    print()


    pickle.dump(model_type,open(save_path+"/"+model_name+".pickle","wb"))