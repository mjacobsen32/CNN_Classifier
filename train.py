

# ------------ Train Function ------------
def train(args, model, device, train_loader, optimizer, epoch, loss_fn, target_count, running_loss, correct, writer):
    model.train() 
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        for i in target:
          target_count[i.item()] += 1
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = output.argmax(dim=1, keepdim=True)
        correct += predicted.eq(target.view_as(predicted)).sum().item()
        if (batch_idx + 1) % args.write_log_interval == 0:
          writer.add_scalar('training loss', running_loss / len(target), epoch * len(train_loader) + batch_idx)
          running_accuracy = correct / (args.write_log_interval * args.batch_size)
          writer.add_scalar('training accuracy', running_accuracy, epoch * len(train_loader) + batch_idx)
          running_loss = 0.0
          correct = 0

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            sys.stdout.flush()
            os.fsync(sys.stdout)
            if args.dry_run:
                break
    return sum(losses) / len(losses)