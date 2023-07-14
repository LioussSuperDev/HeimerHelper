import sys
import os
from os.path import isfile, join
import json
sys.path.insert(0, '..')
from utils import UGGApi
import time

def load_player_data(region, summonerName, save=False, player_save_directory_path=None, matches_save_directory_path=None, force_download=False, Debug=False, max_number_of_matches=3, verbose=False):


    player_neighbour=[]
    
    if verbose:
        print("Doing summoner",summonerName)

    #Handling matches from current player
    
    matches = UGGApi.get_player_match_history(summonerName, regionId=region)
    if verbose:
        print("matches summaries loaded")
    
    returned_matches = []
    for match in matches["matchSummaries"][:max_number_of_matches]:

        masteries_dict = {}

        match_creation_time = match["matchCreationTime"]
        match_id = str(match["matchId"])

        if verbose:
            print("Gathering",match_id,"data")

        #Load or Download match
        local_loaded = False
        if matches_save_directory_path != None:
            maybe_match_file = join(matches_save_directory_path, match_id+".json")
        else:
            maybe_match_file = "UNKNOWN.not.exists"
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
                masteries_dict[p["summonerName"]],p["last10matches"] = get_single_player_stats(region, p["summonerName"], match_creation_time, player_save_directory_path=player_save_directory_path, save=save, verbose=verbose)
            for p in match["matchSummary"]["teamB"]:
                player_neighbour.append(p["summonerName"])
                masteries_dict[p["summonerName"]],p["last10matches"] = get_single_player_stats(region, p["summonerName"], match_creation_time, player_save_directory_path=player_save_directory_path, save=save, verbose=verbose)
            if verbose:
                print("last 10 matches of each player loaded                           ")
        #Saving Match
        if matches_save_directory_path != None and save and not local_loaded:
            file_path = os.path.join(os.path.dirname(__file__), matches_save_directory_path+"\\"+match_id)+".json"
            os.makedirs(matches_save_directory_path, exist_ok=True)
            if(not os.path.isfile(file_path)):
                with open(file_path, "w") as f:
                    f.write(json.dumps(match))
        teamnb = 1
        for p in match["matchSummary"]["teamB"]:
            if p["summonerName"].lower() == summonerName.lower():
                teamnb = 2
        returned_matches.append((match_id,match,masteries_dict,teamnb))

    return player_neighbour,returned_matches

def get_single_player_stats(region, summonerName, matchCreationTime, player_save_directory_path=None, save=False, verbose=False):
    
    page = 1
    stats = []

    if save:
        os.makedirs(player_save_directory_path, exist_ok=True)
        file_path = os.path.join(os.path.dirname(__file__), player_save_directory_path+"\\"+summonerName+".json")
    
    if verbose:
        print("doing",summonerName,"masteries                            ",end="\r")
    masteries = UGGApi.get_player_stats(summonerName, regionId=region)
    if not save or (not os.path.isfile(file_path)):
        #Saving player if can
        if player_save_directory_path != None and save:
            with open(file_path, "w") as f:
                f.write(json.dumps(masteries))


    while len(stats) < 10:
        if verbose:
            print("checking",summonerName,"last 10 matches ( Page",page,")                  ",end="\r")
        matches = UGGApi.get_player_match_history(summonerName, regionId=region, page=page)
        if not (matches != None and "totalNumMatches" in matches and matches["totalNumMatches"] != 0):
            break
        for match in matches["matchSummaries"]:
            if len(stats) >= 10:
                break
            local_creation_time = match["matchCreationTime"]
            if matchCreationTime == -1 or local_creation_time < matchCreationTime:
                stats.append({
                    "win": match["win"],
                    "match": match,
                })
        page += 1
    return masteries,stats


def gather_live_game(regionId, summonerName):
    team_nb = 0

    def role_to_int(role):
        if role == "top":
            return 4
        elif role == "jungle":
            return 1
        elif role == "mid":
            return 5
        elif role == "adc":
            return 3
        elif role == "supp":
            return 2

    game = UGGApi.get_player_current_game(summonerName, regionId=regionId)
    returned_game = {}
    returned_game["matchSummary"] = {}
    returned_game["allPlayerRanks"] = []
    returned_game["matchSummary"]["matchCreationTime"] = int(time.time()*1000)
    masteries = {}
    for team in ["teamA","teamB"]:
        returned_game["matchSummary"][team] = game[team]
        for player in returned_game["matchSummary"][team]:
            if team == "teamA" and player["summonerName"].lower() == summonerName.lower():
                team_nb = 1
            if team == "teamB" and player["summonerName"].lower() == summonerName.lower():
                team_nb = 2

            masteries[player["summonerName"]],player["last10matches"] = get_single_player_stats(regionId, player["summonerName"], returned_game["matchSummary"]["matchCreationTime"])
            returned_game["allPlayerRanks"].append({"summonerName":player["summonerName"],"rankScores":([player["currentSeasonRankScore"]] if player["currentSeasonRankScore"] != None else []) })
            player["role"] = role_to_int(player["currentRole"])

    return returned_game,masteries,team_nb