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
    return json.loads(x.text)["data"]["updatePlayerProfile"]["success"]

def get_player_stats(summonerName, role="all", seasonId=20, queueType=[420], regionId="euw1"):
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
            "query": "query getPlayerStats($queueType: [Int!], $regionId: String!, $role: Int!, $seasonId: Int!, $summonerName: String!) {\n  fetchPlayerStatistics(\n    queueType: $queueType\n    summonerName: $summonerName\n    regionId: $regionId\n    role: $role\n    seasonId: $seasonId\n  ) {\n    basicChampionPerformances {\n      assists\n      championId\n      cs\n      damage\n      damageTaken\n      deaths\n      doubleKills\n      gold\n      kills\n      maxDeaths\n      maxKills\n      pentaKills\n      quadraKills\n      totalMatches\n      tripleKills\n      wins\n      lpAvg\n\n    }\n    exodiaUuid\n    puuid\n    queueType\n    regionId\n    role\n    seasonId\n    __typename\n  }\n}"
           }
    

    x = requests.post("https://u.gg/api",json=data,headers=headers)
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
            "query": "query match($matchId: String!, $regionId: String!, $summonerName: String!, $version: String!) {  match(    matchId: $matchId    regionId: $regionId    summonerName: $summonerName    version: $version  ) {    allPlayerRanks {      rankScores {        lastUpdatedAt        losses        lp        queueType        rank        role        seasonId        tier        wins        __typename      }      exodiaUuid      summonerName      __typename    }    historicalData {      xpDifferenceFrames {        oppValue        timestamp        youValue        __typename      }      teamOneOverview {        bans        baronKills        dragonKills        gold        inhibitorKills        kills        riftHeraldKills        towerKills        __typename      }      teamTwoOverview {        bans        baronKills        dragonKills        gold        inhibitorKills        kills        riftHeraldKills        towerKills        __typename      }      runes      skillPath      statShards      accountIdV3      csDifferenceFrames {        oppValue        timestamp        youValue        __typename      }      finishedItems {        itemId        timestamp        type        __typename      }      goldDifferenceFrames {        oppValue        timestamp        youValue        __typename      }      itemPath {        itemId        timestamp        type        __typename      }      matchId      postGameData {        assists        carryPercentage        championId        cs        damage        deaths        gold        items        jungleCs        keystone        kills        level        role        subStyle        summonerName        summonerSpells        teamId        wardsPlaced        level        __typename      }      primaryStyle      queueType      regionId      subStyle      summonerName      __typename    }    matchSummary {      assists      championId      cs      damage      deaths      gold      items      jungleCs      killParticipation      kills      level      matchCreationTime      matchDuration      matchId      maximumKillStreak      primaryStyle      queueType      regionId      role      runes      subStyle      summonerName      summonerSpells      psHardCarry      psTeamPlay      lpInfo {        lp        placement        promoProgress        promoTarget        promotedTo {          tier          rank          __typename        }        __typename      }      teamA {        championId        summonerName        teamId        role        hardCarry        teamplay        __typename      }      teamB {        championId        summonerName        teamId        role        hardCarry        teamplay        __typename      }      version      visionScore      win      __typename    }    playerInfo {      accountIdV3      accountIdV4      exodiaUuid      iconId      puuidV4      regionId      summonerIdV3      summonerIdV4      summonerLevel      summonerName      __typename    }    playerRank {      exodiaUuid      rankScores {        lastUpdatedAt        losses        lp        queueType        rank        role        seasonId        tier        wins        __typename      }      __typename    }    playerStatistics {      basicChampionPerformances {        assists        championId        cs        damage        damageTaken        deaths        doubleKills        gold        kills        maxDeaths        maxKills        pentaKills        quadraKills        totalMatches        tripleKills        wins        __typename      }      exodiaUuid      puuid      queueType      regionId      role      seasonId      __typename    }    performanceScore {      hardCarry      teamplay      summonerName      __typename    }    winningTeam    __typename  }}"
           }
    x = requests.post("https://u.gg/api",json=data,headers=headers)
    return json.loads(x.text)["data"]["match"]


def get_player_match_history(summonerName, role=[], regionId="euw1", duoName="", championId=[], queueType=[420], seasonIds=[19,20], page=1):
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
    return json.loads(x.text)["data"]["fetchPlayerMatchSummaries"]


def get_player_current_game(summonerName, regionId="euw"):
    data = {
            "operationName": "GetLiveGame",
            "variables": {
                "regionId": regionId,
                "summonerName":summonerName,
            },
            "query": "query GetLiveGame($regionId: String!, $summonerName: String!) {  getLiveGame(regionId: $regionId, summonerName: $summonerName) {    gameLengthSeconds    gameType    queueId    teamA {      banId      championId      championLosses      championWins      championStats {        kills        deaths        assists        __typename      }      currentRole      onRole      partyNumber      previousSeasonRankScore {        lastUpdatedAt        losses        lp        promoProgress        queueType        rank        role        seasonId        tier        wins        __typename      }      currentSeasonRankScore {        lastUpdatedAt        losses        lp        promoProgress        queueType        rank        role        seasonId        tier        wins        __typename      }      roleDatas {        games        roleName        wins        __typename      }      summonerIconId      summonerName      summonerRuneA      summonerRuneB      summonerRuneData      summonerSpellA      summonerSpellB      threatLevel      __typename    }    teamB {      banId      championId      championLosses      championWins      championStats {        kills        deaths        assists        __typename      }      currentRole      onRole      partyNumber      previousSeasonRankScore {        lastUpdatedAt        losses        lp        promoProgress        queueType        rank        role        seasonId        tier        wins        __typename      }      currentSeasonRankScore {        lastUpdatedAt        losses        lp        promoProgress        queueType        rank        role        seasonId        tier        wins        __typename      }      roleDatas {        games        roleName        wins        __typename      }      summonerIconId      summonerName      summonerRuneA      summonerRuneB      summonerRuneData      summonerSpellA      summonerSpellB      threatLevel      __typename    }    __typename  }}"
           }
    x = requests.post("https://u.gg/api",json=data,headers=headers)
    return json.loads(x.text)["data"]["getLiveGame"]
