import argparse
import main
import glob
import os
import pickle
import pandas as pd
import torch
from torch import autograd

from helpers import add_argument, load_dict
import defaults
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Parser for the evaluator 
parser = argparse.ArgumentParser()
evals = parser.add_argument_group('evals')
add_argument(evals, 'storedir', f'checkpoints_{defaults.DATE}', 'Folder of experiments')
evals.add_argument('--epoch', nargs='+', default=None, type=int)
global_args = vars(parser.parse_args())
STORE_DIR = global_args['storedir']

# Find all experiment directories
arg_files = glob.glob(os.path.join(STORE_DIR, '**', defaults.ARG_FILE), recursive=True)
exp_dirs = [os.path.dirname(f) for f in arg_files]
exp_name = [os.path.relpath(exp, STORE_DIR) for exp in exp_dirs]
exp_args = [load_dict(argfile) for argfile in arg_files]

# Sort through the arguments that are disimliar
df = pd.DataFrame()
for n,a in zip(exp_name, exp_args):
    a['exp_folder'] = n
    df = df.append(a, ignore_index=True)
in_vars = [col for col in df.columns if len(pd.unique(df[col])) != 1]

print(">> Experiments in", STORE_DIR)
for i in in_vars: del a[i]
print(pd.DataFrame(a.items(), columns=['Constant', 'Value']))
print()
print(">> Variables:")
print(df[in_vars])

exit()

outputs = ['test_accuracy','adv_accuracy']

epoch = 14
df = pd.DataFrame()

for exp_dir in exp_dirs:
    kwargs = helpers.load_dict(os.path.join(exp_dir, defaults.ARG_FILE))
    args = main.config(**global_args, **eval_args)
    (train_loader, test_loader), device = main.load(args)
    models, _ = main.init(args, device, shape=train_loader.shape)
    checkpoints = [os.path.join(exp_dir, f'save{epoch:03d}.pt'),
                   os.path.join(exp_dir, f'save_perturb{epoch:03d}.pt')]

    try:
        for m,c in zip(models, checkpoints):
            m.load_state_dict(torch.load(c))
    except:
        print("files missing")
        continue
                

    for eps in [0.01,0.05,0.1,0.5,1.0]:
        model, perturb = models
        model.to(device)
        perturb.to(device)
        args.adv_epsilon=eps
        args.adv_norm='infty'
        out = main.test_adv(-1, args, model, device, test_loader)
        df = df.append(dict(eps=eps, **{k:kwargs[k] for k in variables},
                            **{k:out[k] for k in outputs}),
                      ignore_index=True)
    df.to_pickle(os.path.join(args.storedir, 'eval.pkl'))




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
