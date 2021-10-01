def get_model(model_name, num_classes, device, dims):
  print("Input dimensions: {} X {}".format(dims, dims))
  print("Output dimension: {}\n".format(num_classes))
  print("Model: {}".format(model_name))
  if model_name == "Simple":
    nn = NeuralNetwork(num_classes, dims).to(device)
  elif model_name == "LeNet":
    nn = LeNet(num_classes, dims).to(device)
  elif model_name == "Net":
    nn = Net().to(device)
  elif model_name == "BinaryNet":
    nn = BinaryNet(num_classes, dims).to(device)
  elif model_name == "AlexNet":
    nn = AlexNet(num_classes, dims).to(device)
  elif model_name == "SimpleNet":
    nn = SimpleNet(num_classes, dims).to(device)
  elif model_name == "Jacomatt64":
    nn = Jacomatt64(num_classes, dims).to(device)
  elif model_name == "Jacomatt":
    nn = Jacomatt(num_classes, dims).to(device)
  elif model_name == "DeepJacomatt":
    nn = DeepJacomatt(num_classes, dims).to(device)
  elif model_name == "ConvNetSimple":
    nn = ConvNetSimple(num_classes, dims).to(device)
  elif model_name == "UNet":
    nn = UNet(1,num_classes).to(device)
  print(nn)
  return(nn)