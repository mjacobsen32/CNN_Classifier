from neural_nets.AlexNet import AlexNet
import torch.optim as optim
import random
import re
import math
import numpy as np
import PIL
import torch
import imgaug
import csv
import constants as c
from collections import Counter


def get_model(model_name, num_classes, device, dims, output_file):
    output = ""
    output += "Input dimensions: {} X {}\n".format(dims, dims)
    output += "Output dimension: {}\n".format(num_classes)
    output += "Model: {}\n".format(model_name)
    if model_name == "AlexNet":
        nn = AlexNet(num_classes, dims).to(device)
    output += (str(nn)+'\n')
    write_to_file(information=output, output_file_name=output_file)
    return(nn)


def get_optimizer(opt_name, lr, params, weight_decay, output_file):
    output = ""
    output += "Optimizer: {}+'\n'".format(opt_name)
    output += "Learning rate: {}\n".format(lr)
    write_to_file(information=output, output_file_name=output_file)
    if opt_name == "AdaDelta":
        return(optim.Adadelta(params))
    elif opt_name == "Adam":
        return(optim.Adam(params))
    elif opt_name == "SGD":
        return(optim.SGD(params, lr))
    elif opt_name == "RProp":
        return(optim.Rprop(params))
    elif opt_name == "AdaGrad":
        return(optim.Adagrad(params))


def get_perimeter_pixels(image):
    perimeter_pixels = []
    x, y = image.shape
    for (count, i) in enumerate(image):
        for j, j_val in enumerate(i):
            if(count == 0) or (count == x-1) or (j == 0) or (j == y-1):
                perimeter_pixels.append(j_val)
    return(perimeter_pixels)  


def get_random_subset(desired_length, ds_length):
    return(random.sample(range(0, ds_length), desired_length))


def get_sub_dir(image_name):
    cut = re.split("_", image_name)[0:2]
    return(str(cut[0] + "_" + cut[1]))


def print_image_processing(image_preprocessing, output_file):
    output = ""
    output += "Image preprocessing: \n"
    for i in image_preprocessing:
        output += (i + '\n')
    write_to_file(information=output, output_file_name=output_file)


def get_class_weights(class_list, tot_classes, train_len):
    weight_list = []
    for i, val in class_list:
        weight_list.append((i, 1-(val)/int(train_len)))
    final_list = tot_classes * [0]
    for i, val in weight_list:
        final_list[i] = val
    return(final_list)


def get_class_count(indices, tot_classes):
    total_list = []
    data = list(csv.reader(open(c.complete_csv)))
    for i in indices:
        total_list.append(data[i][1])
    full_class_list = Counter(total_list)
    if tot_classes == 2:
        non_det = 0
        class_list = 2 * [0]
        for i in full_class_list:
            if i == str(65):
                class_list[0] = (0, full_class_list[i])
            else:
                non_det += full_class_list[i]
        class_list[1] = (1, non_det)
    else:
        class_list = 90 * [0]
        for count, i in enumerate(full_class_list):
            if i == count:
                class_list[count] = full_class_list[i]
            else:
                class_list[count] = 0
    return(class_list)


def write_to_file(information, output_file_name):
    print(information)
    f = open(output_file_name+'.txt', "a")
    f.write(information)
    f.close()


def split(indices, ratio):
    train_len = math.floor(len(indices) * ratio)
    return(indices[0:train_len], indices[train_len:len(indices)])


def get_background(perimeter_pixels):
    i = np.random.choice(a=perimeter_pixels, size=(224, 224))
    return(i)


def pad_to_224(image):
    perimeter_pixels = get_perimeter_pixels(image)
    background = get_background(perimeter_pixels)
    background_image = PIL.Image.fromarray(np.uint8(background), 'L')
    image_pil = PIL.Image.fromarray(np.uint8(image), 'L')
    x, y = image_pil.size
    x = int(x/2)
    y = int(y/2)
    offset = (112 - x, 112 - y)
    background_image.paste(image_pil, offset)
    return np.asarray(background_image, dtype='uint8')


def get_lr_sched(sched_name, optimizer, gamma, output_file):
    output = ""
    if sched_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=2, verbose=True
            )
    elif sched_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, 
                                                    gamma=gamma, verbose=False)
    output += str(scheduler) + '\n'
    write_to_file(information=output, output_file_name=output_file)
    return(scheduler)


def get_loss_fn(loss_name, class_weights,output_file):
    output = ""
    output += "Loss function: {}+'\n'".format(loss_name)
    output += "class weights: {}+'\n'".format(class_weights)
    write_to_file(output, output_file)
    if loss_name == "nll_loss":
        return(torch.functional.nll_loss)
    elif loss_name == "MSELoss":
        return(torch.nn.MSELoss())
    elif loss_name == "CEL":
        return(torch.nn.CrossEntropyLoss(weight=class_weights,
                                         reduction='sum'))
    elif loss_name == "BCE":
        return(torch.nn.BCELoss())
    elif loss_name == "BCEL":
        return(torch.nn.BCEWithLogitsLoss())


def get_image(image):
    padded_image = pad_to_224(np.array(image, dtype='uint8'))
    seq3 = imgaug.augmenters.Sequential([
            imgaug.augmenters.pillike.FilterSmooth(),
            imgaug.augmenters.pillike.FilterEdgeEnhance(),
            imgaug.augmenters.Sharpen()
            ])
    images_aug3 = seq3(images=[padded_image])
    return(np.asarray(images_aug3[0], dtype='float32'))
