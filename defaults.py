import time

DATE = time.strftime('%Y-%m-%d')

ARG_FILE  = 'args.txt'
SAVE_FILES = ['save{args.epoch:03d}.pt',
              'save_perturb{args.epoch:03d}.pt']
STORE_FILE = 'store.pt'
STORE_DIR = f'checkpoints_{DATE}'

LOOP_MSG = 'Epoch: {args.epoch}'
