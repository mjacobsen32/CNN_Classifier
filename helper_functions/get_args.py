
# ------------ Args ------------
def get_args(listed_args):    
  parser = argparse.ArgumentParser(description='PyTorch Phytoplankton Identification CNN')
  parser.parse_args([])
  
  # Training batch size
  parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                      help='input batch size for training (default: 64)')
  # Test batch size
  parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                      help='input batch size for testing (default: 1000)')
  # Number of Epochs
  parser.add_argument('--epochs', type=int, default=14, metavar='N',
                      help='number of epochs to train (default: 14)')
  # Learning Rate
  parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                      help='learning rate (default: 1.0)')
  # Gamma value
  parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                      help='Learning rate step gamma (default: 0.7)')
  # GPU settings
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  # Dry run
  parser.add_argument('--dry-run', action='store_true', default=False,
                      help='quickly check a single pass')
  # Seed
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  # Batches before logging training status
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  # Save model or not
  parser.add_argument('--save-model', action='store_true', default=False,
                      help='For Saving the current Model')
  # Num classses - 2 vs 90
  parser.add_argument('--num-classes', type=int, default=90,
                      help='how many classes would you like the output layer to have')
  # Optimizer to be used on NN
  parser.add_argument('--optimizer', type=str, default="AdaDelta",
                      help='optimizer')
  # neural net architecture to be used
  parser.add_argument('--model', type=str, default="Simple",
                      help='neural net')
  # dimensions of input for neural net
  parser.add_argument('--input-dimension', type=int, default=(28),
                      help='integer value for width and height of input')
  # percent of data to use
  parser.add_argument('--percent-data', type=float, default=(100),
                      help='percent of data to use for speed of testing')
  # loss function
  parser.add_argument('--loss', type=str, default=("nll_loss"),
                      help='loss function')

  # lr scheduler function
  parser.add_argument('--lr-sched', type=str, default=("ReduceLROnPlateau"),
                      help='learning rate function')
  
  # prediction statistics on or off
  parser.add_argument('--prediction-stats', type=bool, default=False,
                      help='display prediction statistics bool')
  
  # prediction statistics on or off
  parser.add_argument('--write-los-per-epoch', type=int, default=100,
                      help='batch number before logging tensorboard')

  # imgaug filters
  parser.add_argument('--augmentations', nargs='+', default=[],
                      help='augmentations list')


  # manually passing in args
  args = parser.parse_args(listed_args)
  return(args)