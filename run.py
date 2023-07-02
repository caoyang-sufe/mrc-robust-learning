# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

import argparse

from configs import BaseConfig

from src.models.baselines import AlbertLargeFinetunedRACE, BertLargeFinetunedRACE
from src.tools.train_tools import train_baselines
from src.tools.easy_tools import load_args

args = load_args(BaseConfig)
print(args.n_epochs)

print(vars(args))
