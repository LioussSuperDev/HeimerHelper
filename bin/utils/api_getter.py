from riotwatcher import LolWatcher, ApiError
import os

_watcher = None

def get_watcher():
    global _watcher
    if _watcher is None:
        _watcher = LolWatcher(get_api_key())
    return _watcher

def get_api_key():
    file_path = os.path.join(os.path.dirname(__file__), "../../API-KEY.txt")
    with open(file_path) as key_file:
        return key_file.readline()
    