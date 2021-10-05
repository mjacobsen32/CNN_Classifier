from collections import Counter
import numpy as np

def tally_classes(train_set, test_set):
    print(train_set.type())
'''
 #train_classes = [ds.label[i].item() for i in train_set.indices]
    #test_classes = [ds.label[i].item() for i in test_set.indices]
    #print(ds.classes)
    #train_classes = [label for _, label in train_set]
    #print(train_classes)
    #print(dict(Counter(train_set.targets)))
    #d = Counter(train_classes)
    #_, target in train_set
    ##print(target)
    #ds_len = len(train_set) + len(test_set)
    #print(train_set.img_labels)
    print(train_set.img_labels)
    for x in train_set.img_labels:
        print(x)
    _, x = train_set.img_labels
    classes_num = Counter(x)
    print(classes_num)

    #train_classes = Counter(train_classes) # if doesn' work: Counter(i.item() for i in train_classes)
    #test_classes = Counter(test_classes)
    #sum_list = [a + b for a, b in zip(train_classes, test_classes)]'''