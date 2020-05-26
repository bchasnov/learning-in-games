import time
# Globals
DATE = time.strftime('%Y-%m-%d')

# Individual experiments
STORE_DIR = f'checkpoints_{DATE}'
ARG_FILE  = 'args.txt'
SAVE_FILES = ['save{args.epoch:03d}.pt',
              'save_perturb{args.epoch:03d}.pt']
STORE_FILE = 'store.pt'

# Groups of experiments
EVAL_FILE = 'eval.pkl'

# User Interface
LOOP_MSG = 'Epoch: {args.epoch}'
