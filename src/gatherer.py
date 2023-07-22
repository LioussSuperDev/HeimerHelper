import sys
import gatherer_utils
import time

sys.path.insert(0, '..')

def populate_with_player(region, seed_player_names):
    index = 0
    checked = []
    while seed_player_names:
        try:
            print('=====')
            new_players,_ = gatherer_utils.load_player_data(region, seed_player_names[0], True, "data\\matches", False, True, verbose=True, max_number_of_matches=2)
            for player in new_players:
                if not player in seed_player_names and not player in checked:
                    seed_player_names.append(player)
                checked.append(seed_player_names[0])
            del seed_player_names[0]
            print(len(seed_player_names),"players remaining...")
            index += 1
        except KeyError:
            print("Unable to do the current game")
        except:
            print("U.GG RATE LIMIT ! WAITING 2 MINUTES")
            time.sleep(60*2)
            



populate_with_player("euw1",["Liouss","Inting Pineapple","Darkk Sasuke","BBS RayZor","Eau distillée"])
