# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

from src.tools.easy_tools import initialize_logger, terminate_logger


def train_baselines(logging_path):
    logger = initialize_logger(filename=logging_path, mode='w')
    terminate_logger(logger)
