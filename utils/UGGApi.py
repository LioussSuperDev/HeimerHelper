import requests
import json

headers = {
            "Accept-Encoding":"gzip, deflate, br",
            "Accept":"*/*",
            "Content-Type": "application/json",
            "Connection": "keep-alive",
            "User-Agent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
          }

roles_map = {
    "support":2,
    "adc":3,
    "midlane":5,
    "jungle":1,
    "top":4,
    "all":7
}


versions = requests.get("https://ddragon.leagueoflegends.com/api/versions.json", headers=headers)
latest = json.loads(versions.text)[0].strip()
ddragon = requests.get(f"https://ddragon.leagueoflegends.com/cdn/{latest}/data/en_US/champion.json", headers=headers)
championJson = json.loads(ddragon.text)["data"]

def update_ugg(summonerName, regionId="euw1"):
    data = {
            "operationName": "UpdatePlayerProfile",
            "variables": {
                "regionId": regionId,
                "summonerName": summonerName
            },
            "query": "query UpdatePlayerProfile($regionId: String!, $summonerName: String!) {  updatePlayerProfile(region_id: $regionId, summoner_name: $summonerName) {    success    errorReason    __typename  }}"
           }
    x = requests.post("https://u.gg/api",json=data,headers=headers)
    if x.status_code != 200:
        raise ConnectionError
    return json.loads(x.text)["data"]["updatePlayerProfile"]["success"]

def get_player_stats(summonerName, role="all", seasonId=21, queueType=[420], regionId="euw1"):
    role = roles_map[role]
    data = {
            "operationName": "getPlayerStats",
            "variables": {
                "summonerName": summonerName,
                "regionId": regionId,
                "role": role,
                "seasonId": seasonId,
                "queueType": queueType
            },
            "query": "query getPlayerStats($queueType: [Int!], $regionId: String!, $role: [Int!], $seasonId: Int!, $summonerName: String!) {\n  fetchPlayerStatistics(\n    queueType: $queueType\n    summonerName: $summonerName\n    regionId: $regionId\n    role: $role\n    seasonId: $seasonId\n  ) {\n    basicChampionPerformances {\n      assists\n      championId\n      cs\n      damage\n      damageTaken\n      deaths\n      gold\n      kills\n      totalMatches\n      wins\n      lpAvg\n    }\n    exodiaUuid\n    puuid\n    queueType\n    regionId\n    role\n    seasonId\n    __typename\n  }\n}"
           }
    

    x = requests.post("https://u.gg/api",json=data,headers=headers)
    if x.status_code != 200:
        raise ConnectionError
    return json.loads(x.text)["data"]["fetchPlayerStatistics"]

def get_match(matchId, summonerName, version, regionId="euw1"):
    data = {
            "operationName": "match",
            "variables": {
                "matchId": matchId,
                "regionId": regionId,
                "version": version,
                "summonerName": summonerName
            },
            "query": "query match($matchId: String!, $regionId: String!, $summonerName: String!, $version: String!) {\n  match(\n    matchId: $matchId\n    regionId: $regionId\n    summonerName: $summonerName\n    version: $version\n  ) {\n    allPlayerRanks {\n      rankScores {\n        lastUpdatedAt\n        losses\n        lp\n        queueType\n        rank\n        role\n        seasonId\n        tier\n        wins\n        }\n      exodiaUuid\n      summonerName\n    }\n      matchSummary {\n      assists\n      championId\n      cs\n      damage\n      deaths\n      gold\n      items\n      jungleCs\n      killParticipation\n      kills\n      level\n      matchCreationTime\n      matchDuration\n      matchId\n      maximumKillStreak\n      primaryStyle\n      queueType\n      regionId\n      role\n      runes\n      subStyle\n      summonerName\n      summonerSpells\n      psHardCarry\n      psTeamPlay\n      lpInfo {\n        lp\n        placement\n        promoProgress\n        promoTarget\n        promotedTo {\n          tier\n          rank\n            }\n        }\n      teamA {\n        championId\n        summonerName\n        teamId\n        role\n        hardCarry\n        teamplay\n        }\n      teamB {\n        championId\n        summonerName\n        teamId\n        role\n        hardCarry\n        teamplay\n        }\n      version\n      visionScore\n      win\n    }\n    playerInfo {\n      accountIdV3\n      accountIdV4\n      exodiaUuid\n      iconId\n      puuidV4\n      regionId\n      summonerIdV3\n      summonerIdV4\n      summonerLevel\n      summonerName\n    }\n    playerRank {\n      exodiaUuid\n      rankScores {\n        lastUpdatedAt\n        losses\n        lp\n        queueType\n        rank\n        role\n        seasonId\n        tier\n        wins\n        }\n    }\n    winningTeam\n    __typename\n  }\n}"
           }
    x = requests.post("https://u.gg/api",json=data,headers=headers)
    if x.status_code != 200:
        raise ConnectionError
    return json.loads(x.text)["data"]["match"]


def get_player_match_history(summonerName, role=[], regionId="euw1", duoName="", championId=[], queueType=[420], seasonIds=[21], page=1):
    data = {
            "operationName": "FetchMatchSummaries",
            "variables": {
                "championId": championId,
                "duoName": duoName,
                "queueType": queueType,
                "regionId": regionId,
                "role":role,
                "seasonIds":seasonIds,
                "summonerName":summonerName,
                "page":page
            },
            "query": "query FetchMatchSummaries($championId: [Int], $page: Int, $queueType: [Int], $duoName: String, $regionId: String!, $role: [Int], $seasonIds: [Int]!, $summonerName: String!) {  fetchPlayerMatchSummaries(    championId: $championId    page: $page    queueType: $queueType    duoName: $duoName    regionId: $regionId    role: $role    seasonIds: $seasonIds    summonerName: $summonerName  ) {    finishedMatchSummaries    totalNumMatches    matchSummaries {      assists      championId      cs      damage      deaths      gold      items      jungleCs      killParticipation      kills      level      matchCreationTime      matchDuration      matchId      maximumKillStreak      primaryStyle      queueType      regionId      role      runes      subStyle      summonerName      summonerSpells      psHardCarry      psTeamPlay      lpInfo {        lp        placement        promoProgress        promoTarget        promotedTo {          tier          rank          __typename        }        __typename      }      teamA {        championId        summonerName        teamId        role        hardCarry        teamplay        __typename      }      teamB {        championId        summonerName        teamId        role        hardCarry        teamplay        __typename      }      version      visionScore      win      __typename    }    __typename  }}"
           }
    x = requests.post("https://u.gg/api",json=data,headers=headers)
    if x.status_code != 200:
        raise ConnectionError
    return json.loads(x.text)["data"]["fetchPlayerMatchSummaries"]


def get_player_current_game(summonerName, regionId="euw1"):
    data = {
            "operationName": "GetLiveGame",
            "variables": {
                "regionId": regionId,
                "summonerName":summonerName,
            },
            "query": "query GetLiveGame($regionId: String!, $summonerName: String!) {  getLiveGame(regionId: $regionId, summonerName: $summonerName) {    gameLengthSeconds    gameType    queueId    teamA {      banId      championId      championLosses      championWins      championStats {        kills        deaths        assists        __typename      }      currentRole      onRole      partyNumber      previousSeasonRankScore {        lastUpdatedAt        losses        lp        promoProgress        queueType        rank        role        seasonId        tier        wins        __typename      }      currentSeasonRankScore {        lastUpdatedAt        losses        lp        promoProgress        queueType        rank        role        seasonId        tier        wins        __typename      }      roleDatas {        games        roleName        wins        __typename      }      summonerIconId      summonerName      summonerRuneA      summonerRuneB      summonerRuneData      summonerSpellA      summonerSpellB      threatLevel      __typename    }    teamB {      banId      championId      championLosses      championWins      championStats {        kills        deaths        assists        __typename      }      currentRole      onRole      partyNumber      previousSeasonRankScore {        lastUpdatedAt        losses        lp        promoProgress        queueType        rank        role        seasonId        tier        wins        __typename      }      currentSeasonRankScore {        lastUpdatedAt        losses        lp        promoProgress        queueType        rank        role        seasonId        tier        wins        __typename      }      roleDatas {        games        roleName        wins        __typename      }      summonerIconId      summonerName      summonerRuneA      summonerRuneB      summonerRuneData      summonerSpellA      summonerSpellB      threatLevel      __typename    }    __typename  }}"
           }
    x = requests.post("https://u.gg/api",json=data,headers=headers)
    if x.status_code != 200:
        raise ConnectionError
    return json.loads(x.text)["data"]["getLiveGame"]
 

def champion_from_id(championId):
    for champion in championJson:
        if championJson[champion]["key"] == str(championId):
            return championJson[champion]["id"]

opggurlcode = None
def update_opgg_url():
    global opggurlcode
    x = requests.get("https://www.op.gg/champions/nasus/runes/top", headers=headers).text.split("https://s-lol-web.op.gg/_next/static/")
    for i in x[1:]:
        code = i.split("/")[0]
        if not code in ["media","css","chunks"] and len(code) > 15:
            opggurlcode = code
            return

update_opgg_url()
def get_champion_winrate(championId, roleId, tier=None):
    global opggurlcode
    champ1 = champion_from_id(championId)
    for k in roles_map:
        if roles_map[k] == roleId:
            role = k
    if role == "toplane":
        role = "top"
    if role == "midlane":
        role = "mid"
    if opggurlcode == None:
        update_opgg_url()
    if tier == None:
        x = requests.get("https://www.op.gg/_next/data/"+opggurlcode+"/en_US/champions/"+champ1+"/counters/"+role+".json?position="+role, headers=headers)
    else:
        x = requests.get("https://www.op.gg/_next/data/"+opggurlcode+"/en_US/champions/"+champ1+"/counters/"+role+".json?tier="+tier+"&position="+role, headers=headers)
    try:
        props = json.loads(x.text)["pageProps"]
    except:
        update_opgg_url()
        if tier == None:
            x = requests.get("https://www.op.gg/_next/data/"+opggurlcode+"/en_US/champions/"+champ1+"/counters/"+role+".json?position="+role, headers=headers)
        else:
            x = requests.get("https://www.op.gg/_next/data/"+opggurlcode+"/en_US/champions/"+champ1+"/counters/"+role+".json?tier="+tier+"&position="+role, headers=headers)
        props = json.loads(x.text)["pageProps"]
    return props