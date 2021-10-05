from torch.utils import data
from helper_functions.write_to_file import write_to_file 
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import datetime
import time
from dataset_class import PhytoplanktonImageDataset
from train import train
from validation import validation
from helper_functions.print_image_processing import print_image_processing
from helper_functions.get_model import get_model
from helper_functions.get_optimizer import get_optimizer
from helper_functions.get_loss_fn import get_loss_fn
from helper_functions.get_lr_sched import get_lr_sched
from helper_functions.tally_classes import tally_classes
from helper_functions.get_class_weights import get_class_weights
from constants import *
from constants import *
import constants as c
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#!unzip "/content/gdrive/My Drive/Colab Notebooks/all_images/Phytoplankton_Images.zip" > /dev/null

# ------------ Driver ------------
def run(args):
    c.output_file_name = str(datetime.datetime)
    c.tb_current_dir = c.tb_log_parent_dir + str(datetime.datetime)
    image_preprocessing = []
    output = ""

    actual_count = args.num_classes * [0]
    pred_count = args.num_classes * [0]
    target_count = args.num_classes * [0]

    running_loss = 0.0
    running_correct = 0

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

    ds = PhytoplanktonImageDataset(annotations_file = complete_csv, 
                                   img_dir = complete_images, 
                                   transform = tf,
                                   target_transform=None,
                                   num_classes = args.num_classes,
                                   percent = args.percent_data,
                                   dims = args.input_dimension)

    output += "Train dataset length: {}\n".format(len(args.train_indices))
    output += "Test dataset length: {}\n".format(len(args.test_indices))

    train_set = torch.utils.data.Subset(ds, args.train_indices)
    test_set = torch.utils.data.Subset(ds, args.test_indices)

    train_loader = torch.utils.data.DataLoader(train_set,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)

    
    writer = SummaryWriter(c.tb_current_dir)
    
    model = get_model(model_name=args.model, num_classes=args.num_classes, device=device, dims=args.input_dimension)
    optimizer = get_optimizer(opt_name=args.optimizer, lr=args.lr, params=model.parameters(), weight_decay=args.gamma)

    class_weights = torch.tensor(get_class_weights(tally_classes(ds, test_loader))).to(device)
    loss_fn = get_loss_fn(args.loss, class_weights, device)

    output += "Multiplicative factor of learning rate decay: {}\n".format(args.gamma)
    scheduler = get_lr_sched(args.lr_sched, optimizer, args.gamma)

    write_to_file(output)
    output = ""

    start_time = time.clock()
    for epoch in range(1, args.epochs + 1):
        mean_loss = train(args, model, device, 
                          train_loader, optimizer, 
                          epoch, loss_fn, running_loss,
                          running_correct, writer)
        validation(model, device, test_loader, loss_fn, 
                   actual_count, pred_count, writer, epoch)
        scheduler.step(mean_loss)
        sys.stdout.flush()
        os.fsync(sys.stdout)
        writer.add_scalar('learning_rate', scheduler.get_lr(), epoch)
    total_time = time.clock() - start_time

    output += "total training/validation time: {}".format(total_time)
    output += "ms per image: {}".format(
        total_time / (args.train_length+args.test_length)
        )
    output += "train target: {}".format(target_count)
    output += "test count: {}".format(actual_count)
    output += "predicted count: {}".format(pred_count)

    write_to_file(output)

    if args.save_model:
        torch.save(model.state_dict(), str(c.output_file_name) + ".pt")
    writer.close()