import time
import copy
import torch
from tqdm.auto import tqdm


def train_model(model, dataloaders, criterion, optimizer, device,
                num_epochs=25, weights_name='weight_save', is_inception=False):

    since = time.time()

    val_acc_history  = []
    loss_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    stats = {}
    epoch_bar = tqdm(range(num_epochs), desc="Training")

    for epoch in epoch_bar:
        tqdm.write('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        tqdm.write('-' * 10)
        epoch_start = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss     = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux1, aux2 = model(inputs)
                        loss = (criterion(outputs, labels)
                                + 0.3 * criterion(aux1, labels)
                                + 0.3 * criterion(aux2, labels))
                    else:
                        outputs = model(inputs)
                        loss    = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc  = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_end  = time.time()
            elapsed    = epoch_end - epoch_start

            tqdm.write('{} Loss: {:.4f} Acc: {:.4f} | Time: {:.1f}s'.format(
                phase, epoch_loss, epoch_acc, elapsed))
            stats[phase] = f'{epoch_loss:.4f}/{epoch_acc:.4f}'
            epoch_bar.set_postfix(train=stats.get('train', '?'),
                                  val=stats.get('val', '?'),
                                  time=f'{elapsed:.1f}s')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc       = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), weights_name + '.pth')
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            if phase == 'train':
                loss_acc_history.append(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history, loss_acc_history


def evaluate_model(model, test_loader, device):
    """Run model on test set and return accuracy."""
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total   += labels.size(0)

    acc = correct / total
    print(f'Test Accuracy: {acc:.4f}')
    return acc