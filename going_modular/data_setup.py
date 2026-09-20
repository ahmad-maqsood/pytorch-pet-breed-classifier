
import torch

NUM_WORKERS = 2
def create_dataloaders(train_data: torch.utils.data.Dataset,
                       test_data: torch.utils.data.Dataset,
                       batch_size: int,
                       num_workers: int=NUM_WORKERS):

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                   shuffle=True, num_workers=num_workers,
                                                   pin_memory=torch.cuda.is_available())
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers,
                                                  pin_memory=torch.cuda.is_available())

    return train_dataloader, test_dataloader
