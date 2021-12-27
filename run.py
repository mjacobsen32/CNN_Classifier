from torch.utils import data
import torch
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
import os
import time
from dataset_class import PhytoplanktonImageDataset
from train import train
from validation import validation
from train2 import train2
from compute_accuracy import compute_accuracy
from helper_functions import print_image_processing
from helper_functions import get_model
from helper_functions import get_optimizer
from helper_functions import get_loss_fn
from helper_functions import get_lr_sched
from helper_functions import write_to_file 
import numpy as np
import constants as c
import matplotlib.pyplot as plt
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# ------------ Driver ------------
def run(args):
    output = ""

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    tf=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.225])
    ])

    ds = PhytoplanktonImageDataset(annotations_file=c.complete_csv, 
                                   img_dir=c.complete_images, 
                                   transform=tf,
                                   target_transform=None,
                                   num_classes=args.num_classes,
                                   percent=args.percent_data,
                                   dims=args.input_dimension,
                                   augs=args.augmentations)
    
    test_ds = PhytoplanktonImageDataset(annotations_file=c.test_csv, 
                                   img_dir=c.test_images, 
                                   transform=tf,
                                   target_transform=None,
                                   num_classes=args.num_classes,
                                   percent=args.percent_data,
                                   dims=args.input_dimension,
                                   augs=args.augmentations)

    output += "Train dataset length: {}\n".format(len(args.train_indices))
    output += "Validation dataset length: {}\n".format(len(args.validation_indices))
    output += "Test dataset length: {}\n".format(len(args.test_indices))

    if args.cross_validation == False:
        train_set = torch.utils.data.Subset(ds, args.train_indices)
        validation_set = torch.utils.data.Subset(ds, args.validation_indices) # try using train ds as test ds
    elif args.cross_validation == True:
        set1 = torch.utils.data.Subset(ds, range(0,20000,1))
        set2 = torch.utils.data.Subset(ds, range(20000,40000,1))
        set3 = torch.utils.data.Subset(ds, range(40000,60000,1))
        set4 = torch.utils.data.Subset(ds, range(60000,80000,1))
        set5 = torch.utils.data.Subset(ds, range(80000,100000,1))
        set_list = [set1,set2,set3,set4,set5]
        validation_set = set_list.pop(0)
        train_set = [item for sublist in set_list for item in sublist]


    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    validation_loader = torch.utils.data.DataLoader(validation_set, **test_kwargs)
    test_loader = torch.utils.data.DataLoader(test_ds, **test_kwargs)

    model = get_model(model_name=args.model, 
                      num_classes=args.num_classes, 
                      device=device, 
                      dims=args.input_dimension, 
                      output_file=args.output_file_name)

    optimizer = get_optimizer(opt_name=args.optimizer, 
                              lr=args.lr, 
                              params=model.parameters(), 
                              weight_decay=args.gamma, 
                              output_file=args.output_file_name)

    class_weights = torch.tensor(args.class_weights).to(device)
    loss_fn = get_loss_fn(args.loss, class_weights, args.output_file_name)

    output += "Multiplicative factor of learning rate decay: {}\n".format(args.gamma)
    scheduler = get_lr_sched(args.lr_sched, optimizer, args.gamma, args.output_file_name, args.step_size)

    write_to_file(output, args.output_file_name)
    output = ""

    train_acc_list = []
    validation_acc_list = []
    loss_list = []

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train2(args, model, device, train_loader, validation_loader, optimizer, epoch, loss_fn, train_acc_list, validation_acc_list, loss_list)
        scheduler.step() # StepLR
        if args.cross_validation == True:
            set_list.append(validation_set)
            validation_set = set_list.pop(0)
            train_set = [item for sublist in set_list for item in sublist]
        train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
        validation_loader = torch.utils.data.DataLoader(validation_set, **test_kwargs)
    total_time = time.time() - start_time
    
    output += ("Train accuracy list: {}\n".format(train_acc_list))
    output += ("Validation accuracy list: {}\n".format(validation_acc_list))
    output += ("Loss list: {}".format(loss_list))
    write_to_file(output, args.output_file_name)
    with torch.set_grad_enabled(False): # save memory during inference
        output = ('Test accuracy: %.2f%%\n' % (compute_accuracy(model, test_loader, device=device)))
        write_to_file(output, args.output_file_name)


    output += "total training/validation time: {}\n".format(total_time)
    output += "ms per image: {}\n".format(
        total_time / (args.train_length+args.validation_length) / args.epochs
        )

    write_to_file(output, args.output_file_name)

    if args.save_model == True:
        torch.save(model.state_dict(), args.output_file_name + ".pt")
