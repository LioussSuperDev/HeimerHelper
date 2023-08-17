import sys
import gatherer_utils
import time
import socket
import requests
from urllib3.exceptions import MaxRetryError,NewConnectionError
sys.path.insert(0, '..')
def populate_with_player(region, seed_player_names):
    while seed_player_names:
        try:
            print('=====')
            new_players,_ = gatherer_utils.load_player_data(region, seed_player_names[0], True, "data\\matches", False, True, verbose=True, max_number_of_matches=2, time_limit_hours=10)
            for player in new_players:
                seed_player_names.append(player)
            del seed_player_names[0]
            print(len(seed_player_names),"players remaining...               ")
        except (ConnectionError, socket.gaierror, MaxRetryError, NewConnectionError, requests.ConnectionError):
            print("U.GG RATE LIMIT ! WAITING 2 MINUTES               ")
            time.sleep(30)
        except BaseException as e:
            print("Missing data in this game. Skipping the player...")
            del seed_player_names[0]


#P E D M CHALL
populate_with_player("euw1",["HNO mbm","Chinese Script","CHADC4RIM","On Saturn Rings","wp0z3q"])

# #B S G
# populate_with_player("euw1",["Marc25xx", "Freeze Porleone", "wuussaa"])

