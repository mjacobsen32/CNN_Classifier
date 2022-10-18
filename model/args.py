import argparse

parser = argparse.ArgumentParser(description='ConvNet model creation arguments')

parser.add_argument(
    '-out', '--output_folder',
    required=True,
    type=str,
    help="String of output folder to save all information of model train and test"
)

parser.add_argument(
    '-c', '--constants',
    required=False,
    type=str,
    default='scatterPlotConstants',
    help="Constants to use for model creation")

parser.add_argument(
    '-b', '--batch_size',
    choices=(16,32,64,128,256),
    required=False,
    type=int,
    default=64,
    help="Train and test batch size")

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

parser.add_argument(
    '-ss', '--step_size',
    required=False,
    type=int,
    default=10,
    help="Epoch at which to adjust learning rate by gamma. Only used if StepLR is used as the learning rate scheduler. Default = 10")

parser.add_argument(
    '-g', '--gamma',
    required=False,
    type=float,
    default=0.1,
    help="Gamma to adjust learning rate at each step size interval. Only used if StepLR is selected as Learning Rate. Default = 0.1")

parser.add_argument(
    '-e', '--epochs',
    required=False,
    type=int,
    default=40,
    help="Number of training epochs. Default = 40")

parser.add_argument(
    '-m', '--model',
    choices=("AlexNet", "GoogLeNet", "ResNet18"),
    required=False,
    type=str,
    default="GoogLeNet",
    help="Model architecture. Options: ('AlexNet', 'GoogLeNet', 'ResNet50'). Default = 'GoogLeNet")

parser.add_argument(
    '-d', '--device',
    required=False,
    type=str,
    default='cuda',
    help="Memory device for model: 'cuda' GPU or 'cpu'. Default = 'cuda'")

parser.add_argument(
    '-sm', '--save_model',
    required=False,
    type=bool,
    default=False,
    help="Save-model as state dict (bool)"
)
