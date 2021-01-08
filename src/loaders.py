import torch
from torchvision import datasets, transforms

def load(args): 
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset, datadir, batch_size, test_batch_size = args.dataset, args.datadir, args.batch_size, args.test_batch_size
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

    return (train_loader, test_loader), device

