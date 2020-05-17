#!/bin/sh
python main.py --lr1=0.1 --lr2=1.0 --perturb_reg=0.00 --save-model --epochs 20
python main.py --lr1=0.1 --lr2=5.0 --perturb_reg=0.00 --save-model --epochs 20
python main.py --lr1=0.1 --lr2=10.0 --perturb_reg=0.00 --save-model --epochs 20
python main.py --lr1=0.1 --lr2=50.0 --perturb_reg=0.00 --save-model --epochs 20
python main.py --lr1=0.1 --lr2=100.0 --perturb_reg=0.00 --save-model --epochs 20
