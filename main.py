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
from ast import literal_eval
import numpy as np
from torch import autograd

import os
from tensorboardX import SummaryWriter

from uuid import uuid4

from tqdm.auto import tqdm
from helpers import add_argument

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

    add_argument(train, 'seed', 1,         'random seed', 'S')
    add_argument(train, 'dataset',         'mnist')
    add_argument(train, 'batch_size',       64, 'input batch size for training', 'N')
    add_argument(train, 'test_batch_size',  1000, 'input batch size for testing', 'N')
    add_argument(train, 'epochs',           20, 'number of epochs to train', 'N')
    add_argument(train, 'lr1',              1.0, 'learning rate for classifier', 'LR')
    add_argument(train, 'lr2',              1.0, 'learning rate for adversary', 'LR')
    add_argument(train, 'gamma1',           0.7, 'learning rate step gamma (per epoch)', 'M')
    add_argument(train, 'gamma2',           0.7, 'learning rate step gamma (per epoch)', 'M')
    add_argument(train, 'perturb_reg',      0.0, 'regularization on adversarial perturbation', 'REG')

    add_argument(run, 'no_cuda',            False, 'disables CUDA training')
    add_argument(run, 'save_model',         True, 'For Saving the current Model')
    add_argument(run, 'log_interval',       10, 'how many batches to wait before logging training status', 'N')
    add_argument(run, 'datadir',           'data')
    add_argument(run, 'storedir',          'checkpoints')
#    add_argument(run, 'resume_epoch',       1, 'Epoch to resume running at')
    add_argument(run, 'log_smooth',         0.5, 'logging smoothness parameter')

    add_argument(test, 'adv_epsilon',       1., 'magnitude of adversarial perturbation')
    add_argument(test, 'adv_norm',         'inf', 'norm of adversarial perturbation', \
                 choices=['abs', 'l2','inf'])
                      
    parser.set_defaults(**kwargs)
    try:
        # hack for detecting a jupyter lab notebook
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
        data, target = batch
        
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
        
        f1_smooth = (1-args.log_smooth)*f1_smooth + args.log_smooth*f1
        f2_smooth = (1-args.log_smooth)*f2_smooth + args.log_smooth*f2
        
        out = {'loss0':f1, 'loss1':f2, 'loss_sum':f1+f2, 'norm_delta':perturb_norm}

        if batch_idx % args.log_interval == 0:
            logger.append(state['iter'], out)
            desc = (f'{args.loop_msg} | Loss: {f1_smooth:6.3f},{f2_smooth:6.3f} | norm(delta):{perturb_norm:8.5f} ||')
            iterator.set_description(desc)
            iterator.refresh()

        state['iter'] += 1

def test(state, args, model, device, test_loader, logger):
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
    out = {'test_accuracy': correct/len(test_loader.dataset)}
    logger.append(state['iter'], out)
    return out

def test_adv(state, args, model, device, test_loader, logger=None):
    model.eval()
    
    iterator = tqdm(enumerate(test_loader), total=len(test_loader))
    correct = 0
    adv_correct = 0
    for idx, (data, target) in iterator:
        data, target = data.to(device), target.to(device)
        perturb = torch.tensor(torch.zeros(*test_loader.shape), requires_grad=True, device=device)
        output = model(data + perturb)
        loss = F.nll_loss(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        Dperturb_loss = autograd.grad(loss, perturb)[0]
        
        if args.adv_norm == 'infty':
            Dperturb_loss = torch.sign(Dperturb_loss)
        elif args.adv_norm == 'l2':
            Dperturb_loss *= torch.norm(torch.ones(*Dperturb_loss.shape))
            Dperturb_loss /= torch.norm(Dperturb_loss)
        elif args.adv_norm == 'noise':
            Dperturb_loss = torch.rand_like(Dperturb_loss) 
        else:
            raise NotImplemented()
        
        adv_data = data + args.adv_epsilon*Dperturb_loss
        adv_output = model(adv_data)
        adv_pred = adv_output.argmax(dim=1, keepdim=True)
        adv_correct += adv_pred.eq(target.view_as(adv_pred)).sum().item()
        
        desc = (f'Test (adv) | Loss: {loss:10.3f}, {correct}/{len(test_loader.dataset)})')
        iterator.set_description(desc)

    accuracy = correct/len(test_loader.dataset)
    adv_accuracy = adv_correct/len(test_loader.dataset)
    out = {'test_accuracy': accuracy, 'adv_accuracy': adv_accuracy, 'adv_data': adv_data}
    if logger: logger.append(state['iter'], out)
    return out
   
class Logger():
    def __init__(self, writer=None):
        self.df = pandas.DataFrame()
        self.writer = writer

    def append(self, iter, other):
        self.df = self.df.append(other, ignore_index=True)
        if self.writer:
            for arg,val in other.items():
                self.writer.add_scalar(arg, val, iter)
    def to_pickle(self, path):
        self.df.to_pickle(path)

def eval(exp_dir, epoch):
    with open(os.path.join(exp_dir, 'args.txt'), 'r') as f:
        kwargs = literal_eval(f.readline())
        args = config(**kwargs)
    logger = Logger()
    print(f"lr1={args.lr1} lr2={args.lr2}")
    (train_loader, test_loader), device = load(args)
    models, optimizers = init(args, device, shape=train_loader.shape)
    state = {"iter": np.nan}
    save_model = os.path.join(exp_dir, f'save{epoch:03d}.pt')
    save_perturb = os.path.join(exp_dir, f'save_perturb{epoch:03d}.pt')
    out = {}
    try:
        models[0].load_state_dict(torch.load(save_model))
        models[1].load_state_dict(torch.load(save_perturb))
        out = test(state, args, models[0], device, test_loader, logger)
        delta = [_ for _ in models[1].parameters()][0]
        img = torchvision.utils.make_grid(delta, normalize=True)
        torchvision.utils.save_image(img, os.path.join(exp_dir, f'perturb{epoch:03d}.png'))
    except:
        print("model not found")
    return dict(lr1=args.lr1, lr2=args.lr2, **out)
    

def main(exp_id=str(uuid4())):
    state = dict(iter=0, start_time=time.time())
    args = config(exp_id=exp_id)

    # try to make the store dir (if it doesn't exist)
    exp_dir = os.path.join(args.storedir, exp_id)
    try:
        os.makedirs(exp_dir)
    except OSError as e:
        print("Directory exists ({e.message})")

    writer = SummaryWriter(exp_dir)
    logger = Logger(writer)
    with open(os.path.join(exp_dir, 'args.txt'), 'w') as file:
        file.write(str(vars(args)))

    (train_loader, test_loader), device = load(args) 
    models, optimizers = init(args, device, shape=train_loader.shape)
    sched1 = StepLR(optimizers[0], step_size=1, gamma=args.gamma1)
    sched2 = StepLR(optimizers[1], step_size=1, gamma=args.gamma2)
    schedulers = [sched1, sched2]
    for epoch in range(1, args.epochs + 1):
        args.loop_msg = f'Epoch: {epoch}'
        train(state, args, models, device, train_loader, optimizers, logger)
        test(state, args, models[0], device, test_loader, logger)
        schedulers[0].step()
        schedulers[1].step()
        logger.to_pickle(os.path.join(args.storedir, exp_id, 'store.pkl'))

        if args.save_model:
            torch.save(models[0].state_dict(), os.path.join(exp_dir, f"save{epoch:03d}.pt"))
            torch.save(models[1].state_dict(), os.path.join(exp_dir, f"save_perturb{epoch:03d}.pt"))


if __name__ == '__main__':
    main()
