import torch
from helper_functions import write_to_file


# ------------ Validation Function ------------
def validation(model, device, test_loader, loss_fn, pred_count, writer, epoch, output_file):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            for i in pred:
                pred_count[i.item()] += 1 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    output = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    write_to_file(output, output_file)
    writer.add_scalar('accuracy', 100. * correct / len(test_loader.dataset), epoch)