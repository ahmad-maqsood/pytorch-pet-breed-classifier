
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm.auto import tqdm

def accuracy_fn(y_pred_label, y_true):
    correct = torch.eq(y_pred_label, y_true).sum().item()
    accuracy = correct/len(y_pred_label)
    return accuracy

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module,
               device: torch.device):

    model.train()
    train_loss, train_acc = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(y_pred, dim=1)
        train_acc += accuracy_fn(y_pred_class, y)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):

    model.eval()
    test_loss, test_acc = 0,0

    with torch.inference_mode():
        for X,y in dataloader:
            X, y = X.to(device), y.to(device)
            test_logits = model(X)
            loss = loss_fn(test_logits, y)
            test_loss += loss.item()

            test_pred_class = torch.argmax(test_logits, dim=1)
            test_acc += accuracy_fn(test_pred_class, y)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          writer: SummaryWriter = None):

    results = {'train_loss': [], 
               'train_acc': [], 
               'test_loss': [], 
               'test_acc': []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           optimizer=optimizer,
                                           loss_fn=loss_fn,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        print(f'Epoch: {epoch+1} || Train Loss: {train_loss:.2f} || Train Accuracy: {(train_acc*100):.2f} || Test Loss: {test_loss:.2f} || Test Accuracy: {(test_acc*100):.2f}')

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

        if writer:
            writer.add_scalars("Loss", {"train": train_loss, "test": test_loss}, global_step=epoch)
            writer.add_scalars("Accuracy", {"train": train_acc, "test": test_acc}, global_step=epoch)

    if writer:
        writer.close()

    return results
