# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

import argparse

from configs import BaseConfig

from src.tools.train_tools import train_baselines
from src.tools.easy_tools import load_args

args = load_args(Config=BaseConfig)

kwargs = dict(args=args,
              baseline_class="AlbertFinetunedRACE",
              data_name="RACE",
              model_name="albert-large-v1",
              ckpt_path=None,
              )
kwargs = dict(args=args,
              baseline_class="BertFinetunedRACE",
              data_name="RACE",
              model_name="bert-base-uncased",
              ckpt_path=None,
              )
train_baselines(**kwargs)
