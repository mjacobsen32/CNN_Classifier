from helper_functions import write_to_file
from compute_accuracy import compute_accuracy
import torch
import numpy as np

def train2(args, model, device, train_loader, 
           validation_loader, optimizer, epoch, 
           loss_fn, train_acc_list, validation_acc_list, 
           loss_list):
    model.train()
    loss = 0.0
    for batch_idx, (features, targets, paths) in enumerate(train_loader):
        
        features = features.to(device)
        targets = targets.to(device)
            
        ### FORWARD AND BACK PROP
        #logits, _ = model(features)
        logits = model(features)
        loss = loss_fn(logits, targets)
        optimizer.zero_grad()
        
        loss.backward()
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        loss_list.append(loss)
        ### LOGGING
        if not batch_idx % 50:
            output = ('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f\n' 
                   %(epoch, args.epochs, batch_idx, 
                     len(train_loader), loss))
            write_to_file(output, args.output_file_name)

        
    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        val_acc = compute_accuracy(model, validation_loader, device=device, test_bool=False, output_file_name=None)
        train_acc = compute_accuracy(model, train_loader, device=device, test_bool=False, output_file_name=None)
        validation_acc_list.append(float(val_acc))
        train_acc_list.append(float(train_acc))
        output = ('Epoch: %03d/%03d | Train: %.3f%% | Validation: %.3f%%\n' % (
              epoch, args.epochs, train_acc, val_acc))
        write_to_file(output, args.output_file_name)
