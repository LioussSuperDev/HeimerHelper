import sys
import os
import json
import time

sys.path.insert(0, '..')
from bin.utils import api_getter
watcher = api_getter.get_watcher()

def populate_with_player(region, seed_player_names):
    
    index = 0

    list_to_check = []
    for seed_player_name in seed_player_names:
        sumDTO = watcher.summoner.by_name(region,seed_player_name)
        list_to_check.append(sumDTO["puuid"])

    while list_to_check:
        print("Doing #"+str(index)+" ::: "+list_to_check[0])
        
        sumDTO = watcher.summoner.by_puuid(region, list_to_check[0])
        new_players = _populate_with_player(region, sumDTO)

        for player in new_players:
            if not player in list_to_check:
                list_to_check.append(player)
        
        del list_to_check[0]
        index += 1

#12 requests max
def _populate_with_player(region, sumDTO):
    puuid = sumDTO["puuid"]
    encryptedID = sumDTO["id"]
    player_to_check = []

    #Get rank infos
    ranks = watcher.league.by_summoner(region, encryptedID)

    #Handling matches from current player
    matches = watcher.match.matchlist_by_puuid(region,puuid,count=10,queue=420)
    matches_with_results = []
    for match_id in matches:
        match = watcher.match.by_id(region, match_id)

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

        #Saving Match
        file_path = os.path.join(os.path.dirname(__file__), "league_dataset\\matches\\"+match_id)+".json"
        os.makedirs("league_dataset\\matches", exist_ok=True)
        if(not os.path.isfile(file_path)):
            with open(file_path, "w") as f:
                f.write(json.dumps(match))


        matches_with_results.append({"match_id": match_id, "win":current_player_winner})

        #Getting list of players to check next
        for m_puuid in match["metadata"]["participants"]:
            if m_puuid != puuid and not os.path.isfile(os.path.join(os.path.dirname(__file__), "league_dataset\\players\\"+m_puuid+".json")):
                player_to_check.append(m_puuid)
    
    #Saving results
    os.makedirs("league_dataset\\players", exist_ok=True)
    file_path = os.path.join(os.path.dirname(__file__), "league_dataset\\players\\"+puuid+".json")
    if not os.path.isfile(file_path):
            with open(file_path, "w") as f:
                f.write(json.dumps({"player":sumDTO,"ranks":ranks,"last_10_matches":matches_with_results}))

    return player_to_check

populate_with_player("euw1",["DzuUwU","Rood Avel","Liouss","Edarek","Laiferenaevi","Eau distillée"])