import collections


from collections import Counter


def tally_classes(ds, train_set, test_set):
    train_classes = [ds.targets[i] for i in train_set.indices]
    test_classes = [ds.targets[i] for i in test_set.indices]
    train_classes = Counter(train_classes) # if doesn' work: Counter(i.item() for i in train_classes)
    test_classes = Counter(test_classes)
    sum_list = [a + b for a, b in zip(train_classes, test_classes)]
    return(sum_list)