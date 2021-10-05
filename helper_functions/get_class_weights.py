def get_class_weights(ds_len, class_list):
    new_list = []
    for i in class_list:
        new_list.append(1-(i/ds_len))
    return(new_list)
