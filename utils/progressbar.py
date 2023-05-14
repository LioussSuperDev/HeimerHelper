def get_progression(current,total,length=20,filled_str="=",empty_str="-"):
    nb = int(length*current/total)
    return "["+(nb*filled_str)+((length-nb)*empty_str)+"]"