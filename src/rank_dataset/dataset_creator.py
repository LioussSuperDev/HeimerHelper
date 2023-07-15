import sys
import os
from os.path import isfile, join
import json
sys.path.insert(0, '../..')
from utils import progressbar
from dataset_utils import handle_match

os.makedirs(join(os.path.dirname(__file__), "dataset"), exist_ok=True)
os.makedirs(join(os.path.dirname(__file__), "dataset/test"), exist_ok=True)
os.makedirs(join(os.path.dirname(__file__), "dataset/train"), exist_ok=True)

match_list = [f for f in os.listdir("..\\data\\matches") if isfile(join("..\\data\\matches", f))]

index = 0
expl = 0
print()
print("Clearing downloaded data and copying to dataset/")
print("Working...",progressbar.get_progression(index,len(match_list),40,filled_str="■",empty_str=":"),str(round(100*index/len(match_list),2))+"%","("+str(index)+"/"+str(len(match_list))+")",end="\r")

for file in match_list:
    match_cleared_datas = handle_match(file)
    if match_cleared_datas != {}:
        expl += 1
        split = "train"
        if expl%8 == 0:
            split = "test"
        with open(join(os.path.dirname(__file__), join("dataset/"+split,file)), "w") as f:
            try:
                f.write(json.dumps(match_cleared_datas))
            except:
                print("Error writing",file)
    else:
        file_path2 = os.path.join(os.path.dirname(__file__), "..\\data\\matches\\"+file)
        os.remove(file_path2)
    index += 1
    print("Working...",progressbar.get_progression(index,len(match_list),40,filled_str="■",empty_str=":"),str(round(100*index/len(match_list),2))+"%","("+str(index)+"/"+str(len(match_list))+")",end="\r")
print()
print("\nDone ! Found exploitable files :",expl)