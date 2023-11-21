import sys
import os
from os.path import isfile, join
import json
import time
sys.path.insert(0, '..')
from utils import UGGApi



def get_previous_tier(tier):
    if tier.lower()[0] == "b":
        return "iron"
    elif tier.lower()[0] == "s":
        return "bronze"
    elif tier.lower()[0] == "g":
        return "silver"
    elif tier.lower()[0] == "p":
        return "gold"
    elif tier.lower()[0] == "e":
        return "platinum"
    elif tier.lower()[0] == "d":
        return "emerald"
    elif tier.lower()[0] == "m":
        return "diamond"
    elif tier.lower()[0] == "g":
        return "master"
    elif tier.lower()[0] == "c":
        return "grandmaster"
    
def get_previous_rank(rank):
    if rank.lower() == "iii":
        return "iv"
    elif rank.lower() == "ii":
        return "iii"
    elif rank.lower() == "i":
        return "ii"
    
def get_previous_tier_rank(tier,rank):
    if rank.lower() == "iv":
        return get_previous_tier(tier),"i"
    return tier,get_previous_rank(rank)

def get_next_tier(tier):
    if tier.lower()[0] == 'b':
        return "silver"
    elif tier.lower()[0] == "s":
        return "gold"
    elif tier.lower()[0] == "g":
        return "platinum"
    elif tier.lower()[0] == "p":
        return "emerald"
    elif tier.lower()[0] == "e":
        return "diamond"
    elif tier.lower()[0] == "d":
        return "master"
    elif tier.lower()[0] == "m":
        return "grandmaster"
    elif tier.lower()[0] == "g":
        return "challenger"
    elif tier.lower()[0] == "i":
        return "bronze"
    
def get_next_rank(rank):
    if rank.lower() == "iii":
        return "ii"
    elif rank.lower() == "ii":
        return "i"
    elif rank.lower() == "iv":
        return "iii"
    
def get_next_tier_rank(tier,rank):
    if rank.lower() == "i":
        return get_next_tier(tier),"i"
    return tier,get_next_rank(rank)




def load_player_data(region, summonerName, save=False, matches_save_directory_path=None, force_download=False, Debug=False, max_number_of_matches=1, verbose=False, small_verbose=False, time_limit_hours=None):

    player_neighbour=[]
    
    if verbose:
        print("Doing summoner",summonerName)

    #Handling matches from current player
    max_page = ((max_number_of_matches-1)//20)+1
    returned_matches = []
    
    count = 0
    for page in range(1,max_page+1):

        remaining_matches = min(20,max_number_of_matches-(page-1)*20)

        matches = UGGApi.get_player_match_history(summonerName, regionId=region, page=page)
        if verbose:
            print("matches summaries loaded")
        
        
        for match in matches["matchSummaries"][:remaining_matches]:
            if small_verbose:
                print(count,max_number_of_matches,end="\r")
            count += 1
            masteries_dict = {}
            masteries_dict_prev_season = {}
            match_creation_time = match["matchCreationTime"]
            now_time = time.time()*1000

            if time_limit_hours != None and now_time - match_creation_time > time_limit_hours*3600000:
                if verbose:
                    print("match creation time exceeded. skipping current player. late of :",int((now_time - match_creation_time)/3600000)-time_limit_hours,"hours")
                break

            match_id = str(match["matchId"])

            if verbose:
                print("Gathering",match_id,"data")

            #Load or Download match
            local_loaded = False
            if matches_save_directory_path != None:
                maybe_match_file = join(matches_save_directory_path, match_id+".json")
            else:
                maybe_match_file = "UNKNOWN.dont.exist"
            if not force_download and isfile(maybe_match_file):
                with open(os.path.join(os.path.dirname(__file__), maybe_match_file)) as f:
                    if Debug:
                        print("Found Match in local data.")
                    match = json.load(f)
                    match,masteries_dict,masteries_dict_prev_season = match["match"],match["masteries_dict"],match["masteries_dict_prev_season"]
                    local_loaded = True
            else:
                match = UGGApi.get_match(match_id, summonerName, match["version"], regionId=region)

            if match == None:
                continue

            if not local_loaded:
                for team in ["teamA","teamB"]:
                    for p in match["matchSummary"][team]:

                        prank = None
                        for r in match["allPlayerRanks"]:
                            if r["summonerName"] == p["summonerName"]:
                                for rk in r["rankScores"]:
                                    if rk["seasonId"] == 21 and rk["queueType"] == "ranked_solo_5x5":
                                        prank = rk
                        player_neighbour.append(p["summonerName"])
                        masteries_dict_prev_season[p["summonerName"]],masteries_dict[p["summonerName"]],p["last10matches"],p["championStats"] = get_single_player_stats(region, p["summonerName"], match_creation_time, verbose=verbose, champion_played=p["championId"], role=p["role"], player_rank_to_update=prank)
                        
                if verbose:
                    print("last 10 matches of each player loaded                           ")
            
            if "playerRank" in match:
                match.pop("playerRank")
            if "playerInfo" in match:
                match.pop("playerInfo")
            if "performanceScore" in match:
                match.pop("performanceScore")
            if "historicalData" in match:
                match.pop("historicalData")

            #Saving Match
            if matches_save_directory_path != None and save and not local_loaded:
                file_path = os.path.join(os.path.dirname(__file__), matches_save_directory_path+"\\"+match_id)+".json"
                os.makedirs(matches_save_directory_path, exist_ok=True)
                if(not os.path.isfile(file_path)):
                    with open(file_path, "w") as f:
                        f.write(json.dumps({"match":match,"masteries_dict":masteries_dict,"masteries_dict_prev_season":masteries_dict_prev_season}))
            teamnb = 1
            for p in match["matchSummary"]["teamB"]:
                if p["summonerName"].lower() == summonerName.lower():
                    teamnb = 2
            returned_matches.append((match_id,match,masteries_dict_prev_season,masteries_dict,teamnb))
    if small_verbose:
        print("")
    return player_neighbour,returned_matches

def get_single_player_stats(region, summonerName, matchCreationTime, verbose=False, update_ugg=True, champion_played=None, player_rank_to_update=None, role=None):

    page = 1
    stats = []
    if player_rank_to_update != None:
        champ_stats = UGGApi.get_champion_winrate(champion_played, role, tier=player_rank_to_update["tier"])
    else:
        champ_stats = UGGApi.get_champion_winrate(champion_played, role)

    #Updating u.gg statistics
    if update_ugg:
        UGGApi.update_ugg(summonerName,region)

    if verbose:
        print("doing",summonerName,"masteries                            ",end="\r")

    #Getting player champion's statistics
    masteries = UGGApi.get_player_stats(summonerName, regionId=region, seasonId=21)
    prev_masteries = UGGApi.get_player_stats(summonerName, regionId=region, seasonId=20)
    #removing duplicates and finding revelant data
    masteries_dict={}
    masteries_dict_prev_season={}
    for queue in masteries:
        if queue["queueType"] != 420 or queue["seasonId"] != 21:
            continue
        for perf in queue["basicChampionPerformances"]:
            if (not perf["championId"] in masteries_dict) or masteries_dict[perf["championId"]]["totalMatches"] < perf["totalMatches"]:
                masteries_dict[perf["championId"]] = perf
    for queue in prev_masteries:
        if queue["queueType"] != 420 or queue["seasonId"] != 20:
            continue
        for perf in queue["basicChampionPerformances"]:
            if (not perf["championId"] in masteries_dict_prev_season) or masteries_dict_prev_season[perf["championId"]]["totalMatches"] < perf["totalMatches"]:
                masteries_dict_prev_season[perf["championId"]] = perf
                masteries_dict_prev_season[perf["championId"]]["assists"] = int(masteries_dict_prev_season[perf["championId"]]["assists"])
                masteries_dict_prev_season[perf["championId"]]["cs"] = int(masteries_dict_prev_season[perf["championId"]]["cs"])
                masteries_dict_prev_season[perf["championId"]]["kills"] = int(masteries_dict_prev_season[perf["championId"]]["kills"])
                masteries_dict_prev_season[perf["championId"]]["deaths"] = int(masteries_dict_prev_season[perf["championId"]]["deaths"])

    #While we don't have the 10 matches before the current match
    while len(stats) < 10:

        if verbose:
            print("checking",summonerName,"last 10 matches ( Page",page,")                  ",end="\r")
        
        #We get the player match history (=20 games per page)
        matches = UGGApi.get_player_match_history(summonerName, regionId=region, page=page)

        #If there are no more matches to find, we break the loop
        if not (matches != None and "totalNumMatches" in matches and matches["totalNumMatches"] != 0):
            break

        #Iteration over all the 20 matches we found
        for match in matches["matchSummaries"]:
            
            #If we have the 10 matches before the game we can leave
            if len(stats) >= 10:
                break

            local_creation_time = match["matchCreationTime"]
            local_matchid = match["matchId"]
            current_champion = match["championId"]

            #If the game is before the current match, we can add it to the list
            if local_creation_time < matchCreationTime and match["matchDuration"]>500:
                stats.append({
                    "win": match["win"],
                    "match": match,
                })
            
            #If the game is after the current match, we use it to update the players statistics to fit with the period the match is in
            elif (local_creation_time >= matchCreationTime) and match["matchDuration"] > 500:
                if current_champion in masteries_dict:
                    #We don't count this game because it is done after the current match
                    masteries_dict[current_champion]["totalMatches"] -= 1
                    if match["win"]:
                        masteries_dict[current_champion]["wins"] -= 1
                    masteries_dict[current_champion]["damage"] -= match["damage"]
                    masteries_dict[current_champion]["assists"] -= match["assists"]
                    masteries_dict[current_champion]["cs"] -= (match["cs"]+match["jungleCs"])
                    masteries_dict[current_champion]["deaths"] -= match["deaths"]
                    masteries_dict[current_champion]["kills"] -= match["kills"]
                    masteries_dict[current_champion]["gold"] -= match["gold"]

                if player_rank_to_update != None: 
                    if match["win"]:
                        player_rank_to_update["wins"] -= 1
                    else:
                        player_rank_to_update["losses"] -= 1

                    #LP won't be really accurate as we are trying to track them, so if possible ignore it in the dataset, take recent games or round it
                    if "lpInfo" in match and match["lpInfo"]["lp"] != None:
                        player_rank_to_update["lp"] -= match["lpInfo"]["lp"]
                        if match["lpInfo"]["promotedTo"]["rank"] != "":
                            if match["lpInfo"]["lp"] > 0:
                                player_rank_to_update["lp"] = 100 + player_rank_to_update["lp"]
                                player_rank_to_update["tier"],player_rank_to_update["rank"] = get_previous_tier_rank(match["lpInfo"]["promotedTo"]["tier"],match["lpInfo"]["promotedTo"]["rank"])
                            elif match["lpInfo"]["lp"] < 0:
                                player_rank_to_update["lp"] = 100 + match["lpInfo"]["lp"]
                                player_rank_to_update["tier"],player_rank_to_update["rank"] = get_next_tier_rank(match["lpInfo"]["promotedTo"]["tier"],match["lpInfo"]["promotedTo"]["rank"])
        #If not enough matches found we check next page on next iteration
        page += 1

    return masteries_dict_prev_season,masteries_dict,stats,champ_stats


def gather_live_game(regionId, summonerName, update_ugg=True, verbose=True):
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
        
    #updating every ugg profile
    if update_ugg:
        print("getting list of players...")
        game = UGGApi.get_player_current_game(summonerName, regionId=regionId)
        for team in ["teamA","teamB"]:
            for player in game[team]:
                if UGGApi.update_ugg(player["summonerName"],regionId):
                    print(player["summonerName"]+" updated")


    print("getting game...")
    game = UGGApi.get_player_current_game(summonerName, regionId=regionId)
    returned_game = {}
    returned_game["matchSummary"] = {}
    returned_game["allPlayerRanks"] = []
    returned_game["matchSummary"]["matchCreationTime"] = int(time.time()*1000)
    masteries = {}
    prev_masteries = {}
    for team in ["teamA","teamB"]:
        returned_game["matchSummary"][team] = game[team]
        for player in returned_game["matchSummary"][team]:
            print(player["summonerName"]+" data downloaded...")
            if team == "teamA" and player["summonerName"].lower() == summonerName.lower():
                team_nb = 1
            if team == "teamB" and player["summonerName"].lower() == summonerName.lower():
                team_nb = 2

            prev_masteries[player["summonerName"]],masteries[player["summonerName"]],player["last10matches"] = get_single_player_stats(regionId, player["summonerName"], returned_game["matchSummary"]["matchCreationTime"])
            returned_game["allPlayerRanks"].append({"summonerName":player["summonerName"],"rankScores":([player["currentSeasonRankScore"]] if player["currentSeasonRankScore"] != None else []) })
            player["role"] = role_to_int(player["currentRole"])

    return returned_game,prev_masteries,masteries,team_nb