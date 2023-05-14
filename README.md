
# HeimerHelper
## What is it ?
**HeimerHelper** is a Python deep learning project created by **Liouss** to predict the chances of winning your League of Legends games ! **This is still in development but coming really soon I hope.**

I'm doing this project to train new skills and learn how to gather data and train deep neural network models. I do not work for Riot Games or anything related to.

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

- Finally activate the environment and install all the required dependencies for the project to work :
	```bash
	conda activate NAME_OF_YOUR_ENV
	conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
	pip install riotwatcher
	```
I'm not currently using pytorch because I'm only gathering data at the moment, but I will use it sooner or later. Now everything should work correctly !

### How to gather data
- First you need to follow instructions at https://developer.riotgames.com/ to get a **Riot API key**. This step is mandatory for almost everything that follows, so take the time to do it. Note that the Riot development API Key expire after 24 hours and has several limitations (for example only 100 requests every 2 minutes).
- Then create a file named "**API-KEY.txt**" at the root of this repo containing your Riot API Key. The scripts you'll launch will use the key you put inside of this file.
- I use "seed players" to explore data from every rank. You can change the names in the call of populate_with_player in the last line of data/gatherer.py.
- Now start the data/gatherer.py file with the following :

 	```bash
	cd src
	python gatherer.py
	```
	You can stop the script at any moment with **Ctrl+C**. The longer you let it work the better it is as you'll gather more data.


### Content of downloaded data

>**data/league_dataset/matches** contains a file per match downloaded with the Riot API (see https://developer.riotgames.com/apis#match-v5/GET_getMatch)

>**data/league_dataset/players** contains a file per player downloaded with the Riot API (see https://developer.riotgames.com/apis#summoner-v4/GET_getByRSOPUUID), merged with a list of match ids from the 10 last Solo/DuoQ from this player and the result (win or not) for each match, and the ranks of every queues corresponding to this player.

### Create a first exploitable dataset
 - Now start the src/rank_dataset/dataset_creator.py file with the following (expecting you to be in the data repository) :
 
 	```bash
	cd rank_dataset
	python dataset_creator.py
	```
	The result of the created dataset will be in **src/rank_dataset/dataset**

### Predict with my custom model (very bad at the moment)
	Start src/rank_dataset/train.py
 	```bash
	python train.py
	```
