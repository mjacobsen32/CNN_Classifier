from helper_functions import write_to_file
from compute_accuracy import compute_accuracy
import torch


def train2(args, model, device, train_loader, optimizer, epoch, loss_fn):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(device)
        targets = targets.to(device)
            
        ### FORWARD AND BACK PROP
        logits, _ = model(features)
        loss = loss_fn(logits, targets)
        optimizer.zero_grad()
        
        loss.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch+1, args.epochs, batch_idx, 
                     len(train_loader), loss))

        
    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        print('Epoch: %03d/%03d | Train: %.3f%%' % (
              epoch+1, args.epochs, 
              compute_accuracy(model, train_loader, device=device)))