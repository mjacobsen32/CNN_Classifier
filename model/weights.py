def get_class_weights(class_list, tot_classes, train_len):
    weight_list = []
    for i, val in class_list:
        weight_list.append((i, 1-(val)/int(train_len)))
    final_list = tot_classes * [0]
    for i, val in weight_list:
        final_list[i] = val
    return(final_list)