import torch
from torchvision import datasets, transforms

def load(dataset, datadir, batch_size, test_batch_size, **kwargs):
    if dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(datadir, train=True, download=True, 
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(datadir, train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=test_batch_size, shuffle=True, drop_last=True, **kwargs)

    # hack to include the shape of the entire batch
    for d, _ in train_loader:
        train_loader.shape = d.shape
        break
    for d, _ in test_loader:
        test_loader.shape = d.shape
        break

    return (train_loader, test_loader)

