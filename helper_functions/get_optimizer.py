import torch.optim as optim

def get_optimizer(opt_name, lr, params, weight_decay):
  print("\nOptimizer: {}".format(opt_name))
  print("Learning rate: {}\n".format(lr))
  if opt_name == "AdaDelta":
    return(optim.Adadelta(params, lr))
  elif opt_name == "Adam":
    return(optim.Adam(params, lr=lr))
  elif opt_name == "SGD":
    return(optim.SGD(params, lr))