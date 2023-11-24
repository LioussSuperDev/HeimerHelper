import sys
import gatherer_utils
import time
import socket
import requests
import random
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
            time.sleep(60+random.randint(0,20))
        # except BaseException as e:
        #     print("Missing data in this game. Skipping the player...")
        #     del seed_player_names[0]

populate_with_player("euw1",["KaisaFa"])
