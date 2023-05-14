import sys
import gatherer_utils

sys.path.insert(0, '..')
from utils import api_getter
watcher = api_getter.get_watcher()

def populate_with_player(region, seed_player_names):
    index = 0

    print("Retrieving seed player's IDs...")
    list_to_check = []
    checked = []
    for seed_player_name in seed_player_names:
        sumDTO = watcher.summoner.by_name(region,seed_player_name)
        list_to_check.append(sumDTO["puuid"])
    print("Done.")
    while list_to_check:
        print("Loading player #"+str(index))
        new_players = gatherer_utils.load_player_data(region, list_to_check[0], True, "data\\players", "data\\matches", False, True)["player_neighbours"]

        for player in new_players:
            if not player in list_to_check and not player in checked:
                list_to_check.append(player)
        checked.append(list_to_check[0])
        del list_to_check[0]
        index += 1



populate_with_player("euw1",["arno chubb"])