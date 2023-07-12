# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

import argparse
from copy import deepcopy


class BaseConfig:
    parser = argparse.ArgumentParser("--")
    parser.add_argument("--device", default="cpu", type=str, help="Device for computation")
    parser.add_argument("--n_epochs", default=3, type=int, help="Number of training epochs")
    parser.add_argument("--train_batch_size", default=2, type=int, help="Size of train batch")
    parser.add_argument("--dev_batch_size", default=32, type=int, help="Size of dev batch")
    parser.add_argument("--test_batch_size", default=32, type=int, help="Size of test batch")
    parser.add_argument("--optimizer", default="Adam", type=str, help="Optimizer in `torch.optim`")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="Keyword argument `lr` of optimizer")
    parser.add_argument("--weight_decay", default=.9, type=float, help="Keyword argument `weight_decay` of optimizer")
    parser.add_argument("--ckpt_cycle", default=1, type=int, help="Save checkpoint every ? epochs")
    parser.add_argument("--lr_step_size", default=1, type=float, help="Keyword argument `step_size` of scheduler")
    parser.add_argument("--lr_multiplier", default=.95, type=float, help="Keyword argument `gamma` of scheduler")
    parser.add_argument("--dropout_rate", default=.1, type=float, help="Keyword argument `p` of dropout")
    parser.add_argument("--max_length", default=512, type=int, help="Keyword argument `max_length` of tokenizer")
