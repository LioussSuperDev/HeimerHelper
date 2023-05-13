import sys
import os
from os.path import isfile, join
import json
sys.path.insert(0, '..')
from bin.utils import progressbar

def _index_from_lane_and_team(lane, team):
    index = 0
    match lane:
        case "JUNGLE":
            index = 1
        case "MIDDLE":
            index = 2
        case "BOTTOM":
            index = 3
        case "UTILITY":
            index = 4
    
    if team == 200:
        index += 5
    return index

def _rank_to_int(tier, rank, lps):
    r = 0
    match tier:
        case "BRONZE":
            r = 4
        case "SILVER":
            r = 8
        case "GOLD":
            r = 12
        case "PLATINUM":
            r = 16
        case "DIAMOND":
            r = 20
        case "MASTER":
            r = 24
        case "GRANDMASTER":
            r = 28
    match rank:
        case "III":
            r += 1
        case "II":
            r += 2
        case "I":
            r += 3
    return r * 100 + lps

def handle_match(file):
    data = {}

    file_path = os.path.join(os.path.dirname(__file__), "league_dataset\\matches\\"+file)
    if not isfile(file_path):
        return data
    with open(file_path,"r") as f:
        try:
            match = json.load(f)
        except:
            return {}
        
        #getting winner
        winner = 1
        for team in match["info"]["teams"]:
            if team["win"] and team["teamId"] == 200:
                winner = 2
        player_list = match["info"]["participants"]
        for p in player_list:
            position = p["teamPosition"]
            
            champion = p["championId"]
            team_id = p["teamId"]
            summoner_level = p["summonerLevel"]

            #filling rank informations
            player_file_path = os.path.join(os.path.dirname(__file__), "league_dataset\\players\\"+p["puuid"]+".json")
            
            if not isfile(player_file_path):
                return {} #Case if missing Player Data
            with open(player_file_path,"r") as f2:
                p_data = json.load(f2)

                points = -1
                winrate = -1
                hotstreak = False
                for rank in p_data["ranks"]:
                    if rank["queueType"] == "RANKED_SOLO_5x5":
                        points = _rank_to_int(rank["tier"], rank["rank"], rank["leaguePoints"])
                        if rank["losses"]+rank["wins"] != 0:
                            winrate = rank["wins"]/(rank["losses"]+rank["wins"])
                        hotstreak = 1 if rank["hotStreak"] else 0

                nb_won = 0
                nb_total = len(p_data["last_10_matches"])
                for match in p_data["last_10_matches"]:
                    if match["win"]:
                        nb_won += 1
                last_10_winrate = nb_won/nb_total
                p_index = str(_index_from_lane_and_team(position,team_id))
                data["winner"] = winner
                data["player_"+p_index+"_champion"] = champion
                data["player_"+p_index+"_points"] = points
                data["player_"+p_index+"_winrate"] = winrate
                data["player_"+p_index+"_summoner_level"] = summoner_level
                data["player_"+p_index+"_last_10_winrate"] = last_10_winrate
                data["player_"+p_index+"_hotstreak"] = hotstreak
                

    return data

os.makedirs("rank_dataset_0", exist_ok=True)

match_list = [f for f in os.listdir("league_dataset\\matches") if isfile(join("league_dataset\\matches", f))]

index = 0
expl = 0
print()
print("Clearing downloaded data and copying to rank_dataset_0/")
print("Working...",progressbar.get_progression(index,len(match_list),40,filled_str="■",empty_str=":"),str(round(100*index/len(match_list),2))+"%","("+str(index)+"/"+str(len(match_list))+")",end="\r")

for file in match_list:
    match_cleared_datas = handle_match(file)
    if match_cleared_datas != {}:
        expl += 1
        file_path = os.path.join(os.path.dirname(__file__), "rank_dataset_0\\"+file)
        with open(join("rank_dataset_0",file), "w") as f:
            try:
                f.write(json.dumps(match_cleared_datas))
            except:
                print("Error writing",file)
    index += 1
    print("Working...",progressbar.get_progression(index,len(match_list),40,filled_str="■",empty_str=":"),str(round(100*index/len(match_list),2))+"%","("+str(index)+"/"+str(len(match_list))+")",end="\r")
print()
print("\nDone ! Found exploitable files :",expl)