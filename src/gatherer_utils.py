import sys
import os
from os.path import isfile, join
import json
sys.path.insert(0, '..')
from utils import UGGApi


def load_player_data(region, summonerName, save=False, player_save_directory_path=None, matches_save_directory_path=None, force_download=False, Debug=False):


    player_neighbour=[]

    print("Doing summoner",summonerName)

    #Handling matches from current player
    
    matches = UGGApi.get_player_match_history(summonerName, regionId=region)
    print("matches summaries loaded")
    

    for match in matches["matchSummaries"][:3]:
        match_creation_time = match["matchCreationTime"]
        match_id = str(match["matchId"])

        print("Gathering",match_id,"data")

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
            match = UGGApi.get_match(match_id, summonerName, match["version"], regionId=region)

        if not local_loaded:
            for p in match["matchSummary"]["teamA"]:
                player_neighbour.append(p["summonerName"])
                p["last10matches"] = get_last_10_matches_results(region, p["summonerName"], match_creation_time, player_save_directory_path=player_save_directory_path, save=save)
            for p in match["matchSummary"]["teamB"]:
                player_neighbour.append(p["summonerName"])
                p["last10matches"] = get_last_10_matches_results(region, p["summonerName"], match_creation_time, player_save_directory_path=player_save_directory_path, save=save)
            print("last 10 matches of each player loaded                           ")
        #Saving Match
        if matches_save_directory_path != None and save and not local_loaded:
            file_path = os.path.join(os.path.dirname(__file__), matches_save_directory_path+"\\"+match_id)+".json"
            os.makedirs(matches_save_directory_path, exist_ok=True)
            if(not os.path.isfile(file_path)):
                with open(file_path, "w") as f:
                    f.write(json.dumps(match))

    return player_neighbour

def get_last_10_matches_results(region, summonerName, matchCreationTime, player_save_directory_path=None, save=False):
    
    page = 1
    stats = []

    os.makedirs(player_save_directory_path, exist_ok=True)
    file_path = os.path.join(os.path.dirname(__file__), player_save_directory_path+"\\"+summonerName+".json")
    
    print("doing",summonerName,"masteries                            ",end="\r")
    if not os.path.isfile(file_path):
        masteries = UGGApi.get_player_stats(summonerName, regionId=region)
        #Saving player if can
        if player_save_directory_path != None and save:
            with open(file_path, "w") as f:
                f.write(json.dumps(masteries))


    while len(stats) < 10:
        print("checking",summonerName,"last 10 matches ( Page",page,")                  ",end="\r")
        matches = UGGApi.get_player_match_history(summonerName, regionId=region, page=page)
        if not (matches != None and "totalNumMatches" in matches and matches["totalNumMatches"] != 0):
            break
        for match in matches["matchSummaries"]:
            if len(stats) >= 10:
                break
            local_creation_time = match["matchCreationTime"]
            if local_creation_time < matchCreationTime:
                stats.append({
                    "win": match["win"],
                    "match": match,
                })
        page += 1
    return stats