from helper_functions import write_to_file


def train(args, model, device, train_loader, optimizer, epoch, loss_fn, running_loss, correct, writer, output_file):
    model.train() 
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = output.argmax(dim=1, keepdim=True)
        correct += predicted.eq(target.view_as(predicted)).sum().item()

        if batch_idx % args.log_interval == 0:
            calc_loss = (100. * batch_idx / len(train_loader))
            output = "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n".format(epoch, batch_idx * len(data), len(train_loader.dataset), calc_loss, loss.item())
            write_to_file(output, output_file)
            writer.add_scalar('loss', calc_loss, epoch)
            if args.dry_run:
                break
    return sum(losses) / len(losses)