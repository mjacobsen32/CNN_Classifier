import math


def split(indices, ratio):
    train_len = math.floor(len(indices) * ratio)
    return(indices[0:train_len], indices[train_len:len(indices)])