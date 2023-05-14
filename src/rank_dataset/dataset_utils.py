import sys
import os
from os.path import isfile, join
import json
sys.path.insert(0, '../..')
from utils import progressbar

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
    r = -1
    match tier:
        case "IRON":
            r = 0
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
            r = 24
        case "CHALLENGER":
            r = 24
    match rank:
        case "III":
            r += 1
        case "II":
            r += 2
        case "I":
            if r != 24:
                r += 3
    return r * 100 + lps

def handle_player(p_data, position, champion, team_id, summoner_level, queueType="RANKED_SOLO_5x5"):
    data = {}
    points = -1
    winrate = -1
    hotstreak = False
    for rank in p_data["ranks"]:
        if rank["queueType"] == queueType:
            points = _rank_to_int(rank["tier"], rank["rank"], rank["leaguePoints"])
            if rank["losses"]+rank["wins"] != 0:
                winrate = rank["wins"]/(rank["losses"]+rank["wins"])
            hotstreak = 1 if rank["hotStreak"] else 0

    nb_won = 0
    nb_total = len(p_data["last_10_matches"])
    for match in p_data["last_10_matches"]:
        if match["win"]:
            nb_won += 1
    last_10_winrate = nb_won/nb_total if nb_total != 0 else 0.5
    p_index = str(_index_from_lane_and_team(position,team_id))
    if "player_"+p_index+"_champion" in data:
        return {}
    
    data["player_"+p_index+"_champion"] = champion
    data["player_"+p_index+"_points"] = points
    data["player_"+p_index+"_winrate"] = winrate
    data["player_"+p_index+"_summoner_level"] = summoner_level
    data["player_"+p_index+"_last_10_winrate"] = last_10_winrate
    data["player_"+p_index+"_hotstreak"] = hotstreak
    return data

def handle_match(match):
    data = {}

    #getting winner
    winner = 1
    for team in match["info"]["teams"]:
        if team["win"] and team["teamId"] == 200:
            winner = 2
    data["winner"] = winner
    player_list = match["info"]["participants"]

    for p in player_list:
        position = p["teamPosition"]
        champion = p["championId"]
        team_id = p["teamId"]
        summoner_level = p["summonerLevel"]

        #filling rank informations
        player_file_path = os.path.join(os.path.dirname(__file__), "..\\data\\players\\"+p["puuid"]+".json")
        
        if not isfile(player_file_path):
            return {} #Case if missing Player Data
        with open(player_file_path,"r") as f2:
            p_data = json.load(f2)
            p_data_dic = handle_player(p_data,position,champion,team_id,summoner_level)
            for key in p_data_dic:
                data[key] = p_data_dic[key]
            
    return data

def load_and_handle_match(file_path):
    if not isfile(file_path):
        return {}
    with open(file_path,"r") as f:
        try:
            return handle_match(json.load(f))
        except:
            return {} 