#!/bin/sh

python main.py --lr1=0.1 --lr2=10 --perturb_reg=0.000001
python main.py --lr1=0.1 --lr2=10 --perturb_reg=0.000005
python main.py --lr1=0.1 --lr2=10 --perturb_reg=0.00001
python main.py --lr1=0.1 --lr2=10 --perturb_reg=0.00005
python main.py --lr1=0.1 --lr2=10 --perturb_reg=0.0001
python main.py --lr1=0.1 --lr2=10 --perturb_reg=0.0005
python main.py --lr1=0.1 --lr2=10 --perturb_reg=0.001
python main.py --lr1=0.1 --lr2=10 --perturb_reg=0.005
