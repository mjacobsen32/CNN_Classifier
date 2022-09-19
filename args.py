import argparse

parser = argparse.ArgumentParser(description='ConvNet model creation arguments')

parser.add_argument(
    '-b', '--batch_size',
    choices=(16,32,64,128,256),
    required=False,
    type=int,
    default=64,
    help="Train and test batch size",)

parser.add_argument(
    '-o', '--optim',
    choices=("Adam"),
    required=False,
    type=str,
    default="Adam",
    help="Optimization function")

parser.add_argument(
    '-l', '--loss',
    choices=("CEL"),
    required=False,
    type=str,
    default="CEL",
    help="Loss function")

parser.add_argument(
    '-lr', '--learning_rate',
    required=False,
    type=int,
    default=0.001,
    help="Learning rate")

parser.add_argument(
    '-s', '--scheduler',
    choices=("ROP", "Step"),
    required=False,
    type=str,
    default="ROP",
    help="Learning rate scheduler. Supported schedulers are ReduceOnPlateau 'ROP' and StepLR 'Step' ")