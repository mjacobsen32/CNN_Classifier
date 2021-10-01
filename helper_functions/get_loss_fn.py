def get_loss_fn(loss_name, class_weights, device):
  print("Loss function: {}".format(loss_name))
  print("class weights: {}".format(class_weights))
  if loss_name == "nll_loss":
    return(F.nll_loss)
  elif loss_name == "MSELoss":
    return(nn.MSELoss())
  elif loss_name == "CEL":
    #return(nn.CrossEntropyLoss())
    return(nn.CrossEntropyLoss(weight=class_weights, reduction='sum'))
  elif loss_name == "BCE":
    return(nn.BCELoss())
  elif loss_name == "BCEL":
    return(nn.BCEWithLogitsLoss())