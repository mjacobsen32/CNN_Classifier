def get_class_weights(class_list, total_datapoints):
    new_list = []
    for i in class_list:
        new_list.append(1-(i/total_datapoints))
    return(new_list)
