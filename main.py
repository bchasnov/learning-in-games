from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class Perturb(nn.Module):
    def __init__(self, shape):
        super(Perturb, self).__init__()
        self.delta = nn.Parameter(torch.zeros(shape))

    def forward(self):
        return self.delta

def config(arg_dict=None):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr1', type=float, default=1.0, metavar='LR',
                        help='learning rate (player 1)')
    parser.add_argument('--gamma1', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (player 1)')
    parser.add_argument('--lr2', type=float, default=1.0, metavar='LR',
                        help='learning rate (player 2)')
    parser.add_argument('--gamma2', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (player 2)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--perturb_reg', default=1)
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--datadir', default='./data')

    args = []
    if arg_dict is not None:
        for arg,val in arg_dict.items():
            args += [f'--{arg}={val}']
    args = parser.parse_args(args)
    args = parser.parse_args([])
    return args

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

def init(args, device, shape):
    model = Net().to(device)
    perturb = Perturb(shape).to(device) 
    opt1 = optim.Adadelta(model.parameters(), lr=args.lr1)
    opt2 = optim.SGD(perturb.parameters(), lr=args.lr2)
    return (model, perturb), (opt1, opt2)


def train(args, models, device, train_loader, optimizers, epoch):
    model,perturb = models
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizers[0].zero_grad()
        delta = perturb()
        output = model(data+delta)
        delta.detach()
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizers[0].step()

        optimizers[1].zero_grad()
        delta = perturb()
        with torch.no_grad():
            output = model(data+delta)
        loss = -F.nll_loss(output, target) + args.perturb_reg*torch.sum(delta*delta)/2
        loss.backward()
        optimizers[1].step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main(args=None):
    args = config(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader, test_loader = load(args.dataset, args.datadir, args.batch_size, args.test_batch_size, **kwargs)
    models, optimizers = init(args, device, shape=train_loader.shape)
    sched1= StepLR(optimizers[0], step_size=1, gamma=args.gamma1)
    sched2= StepLR(optimizers[1], step_size=1, gamma=args.gamma2)
    schedulers = [sched1, sched2]
    for epoch in range(1, args.epochs + 1):
        train(args, models, device, train_loader, optimizers, epoch)
        test(models[0], device, test_loader)
        scheduler[0].step()
        scheduler[1].step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
