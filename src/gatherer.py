import sys
import gatherer_utils

sys.path.insert(0, '..')

def populate_with_player(region, seed_player_names):
    index = 0
    checked = []
    while seed_player_names:
        try:
            new_players = gatherer_utils.load_player_data(region, seed_player_names[0], True, "data\\players", "data\\matches", False, True)
            for player in new_players:
                if not player in seed_player_names and not player in checked:
                    seed_player_names.append(player)
                checked.append(seed_player_names[0])
        except:
            print("Unexpected error while doing player",seed_player_names[0])
        del seed_player_names[0]
        print(len(seed_player_names),"players remaining...")
        index += 1



populate_with_player("euw1",["fisinho","BASILHSKARAS","GigaChad Anti","emsolo","PiPPEN","212 VISION","brexit gaffer","Practice"])
