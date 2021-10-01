def get_lr_sched(sched_name, optimizer, gamma):
  print("Pytorch scheduluer: {}".format(sched_name))
  if sched_name == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)
  elif shed_name == "StepLR":
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma, verbose=False)
  print(scheduler)
  return(scheduler)