
# HeimerHelper
## What is it ?
**HeimerHelper** is a Python deep learning project created by **Liouss** to predict the chances of winning your League of Legends games ! **This is still in development but coming really soon I hope.**

I'm doing this project to train new skills and learn how to gather datas and train deep neural network models. I do not work for Riot Games or anything related to.

## How to gather datas and train your own models ?

### Installing dependencies 

First setup your python environment. I personaly used **Anaconda** to create a conda environment.
You can install Anaconda from https://www.anaconda.com/download .

- Open an Anaconda Prompt, then create your environment with the following :
	> conda create --name NAME_OF_YOUR_ENV
	> *where NAME_OF_YOUR_ENV is the name of your new conda environment*

- Then configure conda to be compatible with your shell :
	> conda init SHELL_NAME
	>*where SHELL_NAME can be powershell, cmd.exe or bash for exemple*

- Finally activate the environment and install all the required dependencies for the project to work :
	> conda activate NAME_OF_YOUR_ENV
	> conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
	> pip install riotwatcher

I'm not curently using pytorch because I'm only gathering datas at the moment, but I will use it sooner or later. Now everything should work correctly !

### How to gather datas
- First you need to follow instructions at https://developer.riotgames.com/ to get a **Riot API key**. This step is mendatory for almost everything that follows, so take the time to do it. Note that the Riot development API Key expire after 24 hours and have several limitations (for exemple only 100 requests every 2 minutes).
 - Then create a file named "**API-KEY.txt**" at the root of this repo containing your Riot API Key. The scripts you'll launch will use the key you put inside of this file.
 - Now start the data/gatherer.py file with the following command :
	 > python data/gatherer.py

	You can stop the script at any moment with **Ctrl+C**. The longer you let it work the better it is as you'll gather more datas.


### Content of downloaded datas

>**data/league_dataset/matches** contains a file per match downloaded with the Riot API (see https://developer.riotgames.com/apis#match-v5/GET_getMatch)

>**data/league_dataset/players** contains a file per player downloaded with the Riot API (see https://developer.riotgames.com/apis#summoner-v4/GET_getByRSOPUUID), merged with a list of match ids from the 10 last Solo/DuoQ from this player and the result (win or not) for each match, and the ranks of every queues corresponding to this player.
