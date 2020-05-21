import argparse
import main
import glob
import os
import pickle
import pandas as pd
import torch
from torch import autograd
import itertools

from helpers import add_argument, load_dict, fstr
import defaults
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Parse arguments for the evalulators 
parser = argparse.ArgumentParser()
evals = parser.add_argument_group('evals')
add_argument(evals, 'storedir', f'checkpoints_{defaults.DATE}', 'Folder of experiments')
evals.add_argument('--epoch', nargs='+', default=None, type=int)
global_args = parser.parse_args()
STORE_DIR = global_args.storedir

# Find all experiment directories
arg_files = glob.glob(os.path.join(STORE_DIR, '**', defaults.ARG_FILE), recursive=True)
exp_dirs = [os.path.dirname(f) for f in arg_files]
exp_names = [os.path.relpath(exp, STORE_DIR) for exp in exp_dirs]
exp_args = [load_dict(argfile) for argfile in arg_files]

if len(arg_files) == 0:
    raise RuntimeError(f"No experiments found in {STORE_DIR}")

# Print arguments that are constant across all experiments
df = pd.DataFrame()
for n,a in zip(exp_names, exp_args):
    #a['exp_folder'] = n
    df = df.append(a, ignore_index=True)
variables = [col for col in df.columns if len(pd.unique(df[col])) != 1]
for i in variables: del a[i]

print(">> Experiments in", STORE_DIR)
print(pd.DataFrame(a.items(), columns=['Constant', 'Value']))
print()
print(">> Variables:")
print(df[variables])

grid_search = dict(adv_epsilon=[0.01,0.05,0.1,0.5,1.0], adv_norm=['infty', 'l2'])


outputs = ['test_accuracy','adv_accuracy']

epoch = 20
df = pd.DataFrame()

for exp_name, exp_dir in zip(exp_names, exp_dirs):
    print(f">> Experiment {exp_name}")
    kwargs = load_dict(os.path.join(exp_dir, defaults.ARG_FILE))
    args = main.config(**kwargs)
    (train_loader, test_loader), device = main.load(args)
    models, _ = main.init(args, device, shape=train_loader.shape)

    checkpoints = [os.path.join(exp_dir, fstr(f, args=args))
        for f in defaults.SAVE_FILES]

    try:
        for m,c in zip(models, checkpoints):
            m.load_state_dict(torch.load(c))
    except:
        print(">> Warning: Files missing for {exp_dir}")
        continue
                
    for values in itertools.product(*grid_search.values()):
        grid = {}
        for k, val in zip(grid_search.keys(), values):
            grid[k] = val
            vars(args)[k] = val

        model, perturb = models
        model.to(device)
        perturb.to(device)
        out = main.test_adv(-1, args, model, device, test_loader)
        df = df.append(dict(**{k:kwargs[k] for k in variables}, # experiment variables 
                            **{k:vars(args)[k] for k in grid_search.keys()}, # evaluation variables
                            **{k:out[k] for k in outputs}), # output
                      ignore_index=True)
    df.to_pickle(os.path.join(STORE_DIR, 'eval.pkl'))


print(df)
exit()
plt.plot(df['eps'], df['adv_accuracy'])
plt.legend(list(df['lr2']))
plt.xscale('log')

ddf = df[df['eps']==.5]
ddf = ddf[ddf['perturb_reg']<=0.0001]
ddf = ddf[ddf['lr2'] <= 40]
plt.plot(ddf['lr2']/ddf['lr1'], ddf['adv_accuracy'],'o') 
plt.plot(ddf['lr2']/ddf['lr1'], ddf['test_accuracy'],'o') 
plt.xlabel('learning rate ratio')
plt.ylabel('adversarial accuracy (eps=0.5)')

plt.plot(ddf['test_accuracy'], ddf['adv_accuracy'],'o') 
plt.xlim([0.985,.995])
plt.ylim([.6,.9])
plt.xlabel('test accuracy')
plt.ylabel('adversarial accuracy')


plt.imshow(out['adv_data'][0,0].cpu().detach(), cmap='Greys')


import torchvision

def show(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')



exp_dir = exp_dirs[4]


for i in range(14):
    args = main.config(**kwargs)
    (m,p),_ = main.init(args, 'cpu', [64,1,28,28])
    savefile = os.path.join(exp_dir, f'save_perturb{i+1:03d}.pt')
    p.load_state_dict(torch.load(savefile))
    delta = [_ for _ in p.parameters()][0]
    show(torchvision.utils.make_grid(delta, normalize=True))
    plt.show()
