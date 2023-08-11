import sys
import os
from os.path import isfile
import json
sys.path.insert(0, '../..')

def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num

def role_to_int(role):
    if role.lower() == "top":
        return 4
    elif role.lower() == "jungle":
        return 1
    elif role.lower() == "midlane":
        return 5
    elif role.lower() == "adc":
        return 3
    elif role.lower() == "support":
        return 2
    
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

    #Loading the file if possible
    file_path = os.path.join(os.path.dirname(__file__), "..\\data\\matches\\"+file)
    if not isfile(file_path):
        return data
    with open(file_path,"r") as f:
        try:
            match = json.load(f)
        except:
            return {}
    players_queues = match["masteries_dict"]
    match = match["match"]

    #Telling who is the winner
    winner = 1
    if include_victory and match["winningTeam"] != 100:
        winner = 0

            

    return _handle_match(match, players_queues, winner, include_victory=True)


def _handle_match(match, players_queues, winner, include_victory=True):

    for player in players_queues:
        pq2 = {}
        for champion in players_queues[player]:
            pq2[str(champion)] = players_queues[player][champion]
        players_queues[player] = pq2

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
                    if str(player["championId"]) in player_queues and player_queues[str(player["championId"])]["totalMatches"] > 0:
                        perf = player_queues[str(player["championId"])]
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
        

def rank_to_int(tier,rank,lp):

    rank_int = 0
    if rank.lower() == "iii":
        rank_int = 1
    elif rank.lower() == "ii":
        rank_int = 2
    elif rank.lower() == "i":
        rank_int = 3
    tier_int = 0
    if tier.lower() == "bronze":
        tier_int = 4
    elif tier.lower() == "silver":
        tier_int = 8
    elif tier.lower() == "gold":
        tier_int = 12
    elif tier.lower() == "platinum":
        tier_int = 16
    elif tier.lower() == "emerald":
        tier_int = 20
    elif tier.lower() == "diamond":
        tier_int = 24
    elif tier.lower() == "master" or tier.lower() == "grandmaster" or tier.lower() == "challenger":
        return 2800 + lp
    return (tier_int + rank_int) * 100 + lp