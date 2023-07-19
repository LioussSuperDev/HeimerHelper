import sys
import os
from os.path import isfile, join
import json
sys.path.insert(0, '../..')
from utils import progressbar


def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num

def int_to_role(role):
    if role == 4:
        return "TOP"
    elif role == 1:
        return "JUNGLE"
    elif role == 5:
        return "MIDLANE"
    elif role == 3:
        return "ADC"
    elif role == 2:
        return "SUPPORT"
    
def handle_match(file, include_victory=True):
    data = {}

    file_path = os.path.join(os.path.dirname(__file__), "..\\data\\matches\\"+file)
    if not isfile(file_path):
        return data
    with open(file_path,"r") as f:
        try:
            match = json.load(f)
        except:
            return {}
        
    winner = 1
    if include_victory and match["winningTeam"] != 100:
        winner = 0

    players_queues = {}

    for team in ["teamA","teamB"]:
        for player in match["matchSummary"][team]:

            file_path = os.path.join(os.path.dirname(__file__), "..\\data\\players\\"+player["summonerName"]+".json")
            if isfile(file_path):
                 with open(file_path,"r") as f:
                    player_queues = json.load(f)
                    players_queues[player["summonerName"]] = player_queues
            

    return _handle_match(match, players_queues, winner, include_victory=True)


def _handle_match(match, players_queues, winner, include_victory=True):
    data = {}
        
    if include_victory:
        data = {"teamA":{},"teamB":{},"winner":winner}
    else:
        data = {"teamA":{},"teamB":{}}

    for team in ["teamA","teamB"]:
        for player in match["matchSummary"][team]:

            pdata = {"matches":[],"wins":0,"losses":0,"premade":[],"championData":{"championId":player["championId"]}}

            if player["summonerName"] in players_queues:
                player_queues = players_queues[player["summonerName"]]
                if player_queues != None:
                    for queue in player_queues:
                        if queue["queueType"] != 420 or queue["seasonId"] != 20:
                            continue
                        for perf in queue["basicChampionPerformances"]:
                            if perf["championId"] == player["championId"]:
                                pdata["championData"]["wins"] = perf["wins"]
                                pdata["championData"]["totalMatches"] = perf["totalMatches"]
                                pdata["championData"]["csPerMatch"] = perf["cs"]/perf["totalMatches"]
                                pdata["championData"]["damagePerMatch"] = perf["damage"]/perf["totalMatches"]
                                pdata["championData"]["deathsPerMatch"] = perf["deaths"]/perf["totalMatches"]
                                pdata["championData"]["killsPerMatch"] = perf["kills"]/perf["totalMatches"]
                                pdata["championData"]["assistsPerMatch"] = perf["assists"]/perf["totalMatches"]
                                pdata["championData"]["goldPerMatch"] = perf["gold"]/perf["totalMatches"]
            

            #adding basic data about player
            for rank_player in match["allPlayerRanks"]:
                if rank_player["summonerName"] == player["summonerName"]:
                    for rank in rank_player["rankScores"]:
                        if rank["queueType"] == "ranked_solo_5x5":
                            pdata["tier"] = rank["tier"]
                            pdata["rank"] = rank["rank"]
                            pdata["lp"] = rank["lp"]
                            pdata["wins"] = rank["wins"]
                            pdata["losses"] = rank["losses"]

            

            for last_match in player["last10matches"]:
                nameInLastMatch = last_match["match"]["summonerName"]
                isInTeam = -1
                #finding premades in previous game
                local_pre = {"teamA":[],"teamB":[]}
                for previous_team in ["teamA","teamB"]:
                    for previous_player in last_match["match"][previous_team]:
                        if previous_player["summonerName"] == nameInLastMatch:
                            isInTeam = previous_team
                        for other_player in match["matchSummary"][team]:
                            if other_player["summonerName"] != nameInLastMatch and other_player["summonerName"] == previous_player["summonerName"]:
                                local_pre[previous_team].append(previous_player["summonerName"])
                # if isInTeam == -1:
                #     continue
                pdata["premade"] += local_pre[isInTeam]

                #adding match to list
                pdata["matches"].append({
                    "win":last_match["win"],
                    "matchDuration":last_match["match"]["matchDuration"],
                    "championId":last_match["match"]["championId"],
                    "kills":last_match["match"]["kills"],
                    "damage":last_match["match"]["damage"],
                    "gold":last_match["match"]["gold"],
                    "deaths":last_match["match"]["deaths"],
                    "assists":last_match["match"]["assists"],
                    "cs":last_match["match"]["cs"],
                    "visionScore":last_match["match"]["visionScore"],
                    "role":int_to_role(last_match["match"]["role"]),
                    "creation_gap":match["matchSummary"]["matchCreationTime"]-last_match["match"]["matchCreationTime"]
                    })
            if len(pdata["premade"]) != 0:
                pdata["premade"] = [most_frequent(pdata["premade"])]
                for premade_player in match["matchSummary"][team]:
                    if premade_player["summonerName"] == pdata["premade"][0]:
                        pdata["premade"] = [int_to_role(premade_player["role"])]
            data[team][int_to_role(player["role"])] = pdata
    return data

def load_and_handle_match(file_path):
    if not isfile(file_path):
        return {}
    with open(file_path,"r") as f:
        try:
            return handle_match(json.load(f))
        except:
            return {} 