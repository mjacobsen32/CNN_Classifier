import sys, os
import torch

# ------------ Validation Function ------------
def validation(model, device, test_loader, loss_fn, actual_count, pred_count):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            for i in target:
              actual_count[i.item()] += 1
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            #test_loss += loss_fn(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            for i in pred:
              pred_count[i.item()] += 1 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    sys.stdout.flush()
    os.fsync(sys.stdout)