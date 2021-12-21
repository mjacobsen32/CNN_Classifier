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

    train_set = torch.utils.data.Subset(ds, args.train_indices)
    validation_set = torch.utils.data.Subset(ds, args.validation_indices) # try using train ds as test ds

    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    validation_loader = torch.utils.data.DataLoader(validation_set, **test_kwargs)
    test_loader = torch.utils.data.DataLoader(test_ds, **test_kwargs)

    writer = SummaryWriter(args.tb_dir)

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
    scheduler = get_lr_sched(args.lr_sched, optimizer, args.gamma, args.output_file_name)

    write_to_file(output, args.output_file_name)
    output = ""

    train_acc_list = []
    validation_acc_list = []
    loss_list = []

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train2(args, model, device, train_loader, validation_loader, optimizer, epoch, loss_fn, train_acc_list, validation_acc_list, loss_list)
        scheduler.step() # StepLR
    total_time = time.time() - start_time

    x = np.array(range(0,args.epochs,1))
    fig, ax1 = plt.subplots() 
  
    ax1.set_xlabel('epochs') 
    ax1.set_ylabel('accuracy', color = 'green') 
    ax1.plot(x, np.array(train_acc_list), color = 'green') 
    ax1.plot(x, np.array(validation_acc_list), color = 'blue')
    ax1.tick_params(axis ='y', labelcolor = 'green') 
  
    # Adding Twin Axes

    ax2 = ax1.twinx() 
  
    ax2.set_ylabel('loss', color = 'red') 
    ax2.plot(x, np.array(loss_list), color = 'red') 
    ax2.tick_params(axis ='y', labelcolor = 'red') 
    plt.savefig(args.start + '_plot.png')

    with torch.set_grad_enabled(False): # save memory during inference
        print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader, device=device)))

    output += "total training/validation time: {}\n".format(total_time)
    output += "ms per image: {}\n".format(
        total_time / (args.train_length+args.validation_length) / args.epochs
        )
    #output += "train class count: {}\n".format(args.train_class_count)
    #output += "test class count: {}\n".format(args.test_class_count)
    #output += "validation class count: {}\n".format(pred_count)

    write_to_file(output, args.output_file_name)

    if args.save_model == True:
        torch.save(model.state_dict(), args.output_file_name + ".pt")
    writer.close()
