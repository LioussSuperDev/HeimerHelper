
# HeimerHelper
## What is it ?
**HeimerHelper** is a Python deep learning project created by **Liouss** to predict the chances of winning your League of Legends games ! **This is still in development but coming really soon I hope.**

I'm doing this project to train new skills and learn how to gather data and train deep neural network models. I do not work for Riot Games or anything related to.

/!\ as I had no time to do clean code (personal project and tests for the moment) please execute every python file while being directly in the folder containing it ! I messed up with the file path system

## How to gather data and train your own models ?

### Installing dependencies 

First setup your python environment. I personally used **Anaconda** to create a conda environment.
You can install Anaconda from https://www.anaconda.com/download .

- Open an Anaconda Prompt, then create your environment with the following :
	```bash
	conda create --name NAME_OF_YOUR_ENV
	```
	> *where NAME_OF_YOUR_ENV is the name of your new conda environment*

- Then configure conda to be compatible with your shell :
 	```bash
	conda init SHELL_NAME
	```
	>*where SHELL_NAME can be powershell, cmd.exe or bash for example*

- Finally activate the environment and install the pytorch dependencies for the project to work. For NVIDIA GPU do :
	```bash
	conda activate NAME_OF_YOUR_ENV
	conda install numpy pandas tensorboard matplotlib tqdm pyyaml -y
	conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
	```
- If you have an AMD GPU, please instead use the following link to create your environment : https://learn.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-windows

### How to gather data to create your own dataset
- I use "seed players" to explore data from every ranks. You can change the names in the call of populate_with_player in the last line of src/gatherer.py.
- The api I'm using is the u.gg api. If you spam the server the algorithm may stop so if you need add a delay between every request
- Now start the data/gatherer.py file with the following :

 	```bash
	cd src
	python gatherer.py
	```
	You can stop the script at any moment with **Ctrl+C**. The longer you let it work the better it is as you'll gather more data.


### Content of downloaded data

>**data/league_dataset/matches** contains a file per match downloaded with the u.gg API

>**data/league_dataset/players** contains a file per player, it contains the masteries of the champions the player plays (comes from u.gg too).

### Create a first exploitable dataset
 - Now start the src/rank_dataset/dataset_creator.py file with the following (expecting you to be in the src repository) :
 
 	```bash
	cd rank_dataset
	python dataset_creator.py
	```
	The result of the created dataset will be in **src/rank_dataset/dataset**

### train with my custom model
	Start src/rank_dataset/train.py
 	```bash
	python train.py
	```
### predict with my custom model
	Start src/rank_dataset/prediction.py
	Please note that you need to specify the models you use for prediction in the file. If you didn't train one you need to.
 	```bash
	python prediction.py
	```