import re

def get_sub_dir(image_name):
    cut = re.split("_", image_name)[0:2]
    return(str(cut[0] + "_" + cut[1]))
