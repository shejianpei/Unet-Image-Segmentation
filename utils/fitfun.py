import torch
import time
import wandb


def fit(epoch, model, trainloader, testloader, optimizer, loss_fn, exp_lr_scheduler, config):
    correct = 0
    total = 0
    running_loss = 0
    model.train()
    i = 0
    for x, y in trainloader:
        i += 1
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
            model = model.to('cuda')
        start_time = time.time()
        y_pred = model(x)
        end_time = time.time()
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
            # print("epoch: ", epoch,
            #       "总次数: ", len(trainloader.dataset)//config.batch_size,
            #       '学习率: ', round(optimizer.param_groups[0]['lr'], 7),
            #       "当前次数: ", i,
            #       "time: ", end_time - start_time,
            #       "当前loss: ", round(loss.item()/config.batch_size, 7),
            #       "总loss: ", round(running_loss, 7),
            #       "当前acc: ", correct / (total * config.SUO_FANG_IMG_SIZE * config.SUO_FANG_IMG_SIZE))
    exp_lr_scheduler.step()
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / (total * config.get("SUO_FANG_IMG_SIZE") * config.get("SUO_FANG_IMG_SIZE"))

    test_correct = 0
    test_total = 0
    test_running_loss = 0

    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for x, y in testloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
                model = model.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / (test_total * config.get("SUO_FANG_IMG_SIZE") * config.get("SUO_FANG_IMG_SIZE"))
    end_time = time.time()
    print('epoch:', epoch,
          'loss:', round(epoch_loss, 5),
          'accuracy:', round(epoch_acc, 5),
          'test_loss:', round(epoch_test_loss, 5),
          'test_accuracy:', round(epoch_test_acc, 5)
          )
    wandb.log({'epoch': epoch,
               'lr': optimizer.param_groups[0]['lr'],
               'loss': epoch_loss,
               'accuracy': epoch_acc,
               'test_loss': epoch_test_loss,
               'test_accuracy': epoch_test_acc})
    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc