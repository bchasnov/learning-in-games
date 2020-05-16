from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from loaders import load
import time
import pandas
import sys

import os
from tensorboardX import SummaryWriter

from uuid import uuid4

if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm

class Net(nn.Module):
    """ default pytorch example """
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
    """ constant perturbation """
    def __init__(self, shape):
        super(Perturb, self).__init__()
        self.delta = nn.Parameter(torch.zeros(shape))

    def forward(self):
        return self.delta

def config(**kwargs):
    # Training settings
    parser = argparse.ArgumentParser(description='learning in games')
    train = parser.add_argument_group('train')
    test = parser.add_argument_group('test')
    run = parser.add_argument_group('run')
    train.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    train.add_argument('--dataset', default='mnist')
    train.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    test.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    train.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    train.add_argument('--lr1', type=float, default=1.0, metavar='LR',
                        help='learning rate (player 1)')
    train.add_argument('--gamma1', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (player 1)')
    train.add_argument('--lr2', type=float, default=1.0, metavar='LR',
                        help='learning rate (player 2)')
    train.add_argument('--gamma2', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (player 2)')
    train.add_argument('--perturb_reg', type=float, default=0.0)
    run.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    run.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    run.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    run.add_argument('--datadir', default='./data')
    run.add_argument('--storedir', default='./checkpoints')
    run.add_argument('--log_smooth', default=0.5)
    parser.set_defaults(**kwargs)
    try:
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            args = parser.parse_args('')
    except:        
        args = parser.parse_args()
    return args


def init(args, device, shape):
    model = Net().to(device)
    perturb = Perturb(shape).to(device) 
    opt1 = optim.SGD(model.parameters(), lr=args.lr1)
    opt2 = optim.SGD(perturb.parameters(), lr=args.lr2)
    return (model, perturb), (opt1, opt2)

def loss(model, delta, batch):
    (data, target) = batch
    output = model(data + delta)
    return F.nll_loss(output, target)

def train(state, args, models, device, loader, optimizers, logger):
    model, perturb = models
    model.train()
    
    iterator = tqdm(enumerate(loader), total=len(loader))
    
    f1_smooth = 0
    f2_smooth = 0
    for batch_idx, (data, target) in iterator:
        batch = data.to(device), target.to(device)
        
        optimizers[0].zero_grad()
        
        delta = perturb()
        f1 = loss(model, delta, batch)
        f1.backward()
        
        optimizers[0].step()
        optimizers[1].zero_grad()
        
        delta = perturb()
        perturb_norm = torch.sum(delta*delta)/2
        f2 = -loss(model, delta, batch) + args.perturb_reg*perturb_norm
        
        f2.backward()
        
        optimizers[1].step()
        
        if batch_idx % args.log_interval == 0:
            pass
#            print('train epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f} norm(delta): {}'.format(
#                epoch, batch_idx * len(data), len(loader.dataset),
#                100. * batch_idx / len(loader), loss.item(), perturb_norm))

        f1_smooth = (1-args.log_smooth)*f1_smooth + args.log_smooth*f1
        f2_smooth = (1-args.log_smooth)*f2_smooth + args.log_smooth*f2
       
        desc = (f'{args.loop_msg} | Loss: {f1_smooth:6.3f},{f2_smooth:6.3f} | norm(delta):{perturb_norm:8.5f} ||')

        if batch_idx % args.log_interval == 0:
            logger.append(state['iter'], {'loss0':f1, 'loss1':f2, 'loss_sum':f1+f2, 'norm_delta':perturb_norm})
        iterator.set_description(desc)
        iterator.refresh()

        state['iter'] += 1

def test(state, model, device, test_loader, logger):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        iterator = tqdm(enumerate(test_loader), total=len(test_loader))
        for idx, (data, target) in iterator:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            desc = (f'Test | Loss: {test_loss:10.3f}, {correct}/{len(test_loader.dataset)}({correct/((idx+1)/test_loader.batch_size*len(test_loader.dataset))}%)')
            iterator.set_description(desc)
            iterator.refresh()

    test_loss /= len(test_loader.dataset)
    logger.append(state['iter'], {'test_accuracy': correct/len(test_loader.dataset)})
   
class Logger():
    def __init__(self, writer):
        self.df = pandas.DataFrame()
        self.writer = writer

    def append(self, iter, other):
        self.df.append(other, ignore_index=True)
        for arg,val in other.items():
            self.writer.add_scalar(arg, val, iter)
    def to_pickle(self, path):
        self.df.to_pickle(path)


def main(exp_id=None):
    state = dict(iter=0, start_time=time.time())
    args = config()

    # make a new experiment id
    if not exp_id:
        exp_id = str(uuid4())

    # try to make the store dir (if it doesn't exist)
    storefile = os.path.join(args.storedir, exp_id)
    try:
        os.makedirs(storefile)
    except OSError as e:
        print("Directory exists ({e.message})")

    writer = SummaryWriter(storefile)
    logger = Logger(writer)
    with open(os.path.join(storefile, 'args.txt'), 'w') as file:
        file.write(str(vars(args)))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader, test_loader = load(args.dataset, args.datadir, args.batch_size, args.test_batch_size, **kwargs)
    models, optimizers = init(args, device, shape=train_loader.shape)
    sched1 = StepLR(optimizers[0], step_size=1, gamma=args.gamma1)
    sched2 = StepLR(optimizers[1], step_size=1, gamma=args.gamma2)
    schedulers = [sched1, sched2]
    for epoch in range(1, args.epochs + 1):
        args.loop_msg = f'Epoch: {epoch}'
        train(state, args, models, device, train_loader, optimizers, logger)
        test(state, models[0], device, test_loader, logger)
        schedulers[0].step()
        schedulers[1].step()
        logger.to_pickle(os.path.join(args.storedir, exp_id, 'store.pkl'))

        if args.save_model:
            torch.save(models[0].state_dict(), os.path.join(storefile, "save{epoch:03d}.pt"))


if __name__ == '__main__':
    main()
