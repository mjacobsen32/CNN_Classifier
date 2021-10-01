import random


def get_random_subset(desired_length, ds_length):
    return(random.sample(range(0, ds_length), desired_length))
