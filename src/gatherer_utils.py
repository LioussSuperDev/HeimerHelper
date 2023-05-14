import sys
import os
from os.path import isfile, join
import json
sys.path.insert(0, '..')
from utils import api_getter


def load_player_data(region, puuid, save=False, player_save_directory_path=None, matches_save_directory_path=None, force_download=False, Debug=False):

    #Trying to load player data from local database
    maybe_player_file = join(player_save_directory_path, puuid+".json")
    if player_save_directory_path != None and not force_download and isfile(maybe_player_file):
        with open(os.path.join(os.path.dirname(__file__), maybe_player_file)) as f:
            if Debug:
                print("Found player in local data.")
            return json.load(f)
        
    watcher = api_getter.get_watcher()
    sumDTO = watcher.summoner.by_puuid(region,puuid)
    puuid = sumDTO["puuid"]
    encryptedID = sumDTO["id"]
    player_neighbours = []
    #Get rank infos
    ranks = watcher.league.by_summoner(region, encryptedID)

    #Handling matches from current player
    matches = watcher.match.matchlist_by_puuid(region,puuid,count=10,queue=420)
    matches_with_results = []
    for match_id in matches:

        #Load or Download match
        local_loaded = False
        maybe_match_file = join(matches_save_directory_path, match_id+".json")
        if not force_download and isfile(maybe_match_file):
            with open(os.path.join(os.path.dirname(__file__), maybe_match_file)) as f:
                if Debug:
                    print("Found Match in local data.")
                match = json.load(f)
                local_loaded = True
        else:
            match = watcher.match.by_id(region, match_id)

        #Saving Match
        if matches_save_directory_path != None and save and not local_loaded:
            file_path = os.path.join(os.path.dirname(__file__), matches_save_directory_path+"\\"+match["metadata"]["matchId"])+".json"
            os.makedirs(matches_save_directory_path, exist_ok=True)
            if(not os.path.isfile(file_path)):
                with open(file_path, "w") as f:
                    f.write(json.dumps(match))

        #Finding winner team
        winner_team = 0
        for t in match["info"]["teams"]:
            if t["win"]:
                winner_team = t["teamId"]

        #Finding if current player is in winner team
        current_player_winner = False
        participants = match["info"]["participants"]
        for p in participants:
            if p["puuid"] == puuid and p["teamId"] == winner_team:
                current_player_winner = True
            player_neighbours.append(p["puuid"])

        matches_with_results.append({"match_id": match_id, "win":current_player_winner})

    #Saving player if can
    if player_save_directory_path != None and save:
        os.makedirs(player_save_directory_path, exist_ok=True)
        file_path = os.path.join(os.path.dirname(__file__), player_save_directory_path+"\\"+puuid+".json")
        if not os.path.isfile(file_path):
                with open(file_path, "w") as f:
                    f.write(json.dumps({"player":sumDTO,"ranks":ranks,"last_10_matches":matches_with_results, "player_neighbours":player_neighbours}))
    return {"player":sumDTO,"ranks":ranks,"last_10_matches":matches_with_results, "player_neighbours":player_neighbours}
