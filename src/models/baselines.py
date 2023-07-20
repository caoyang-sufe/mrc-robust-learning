# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
import os
import re
import torch
from torch.nn import Module, Linear, Dropout, CrossEntropyLoss
from transformers import (AutoTokenizer, AutoModel,
                          AlbertTokenizer, AlbertModel,
                          BertTokenizer, BertModel,
                          RobertaTokenizer, RobertaModel)

from src.models.base import BaseSingleChoiceMRC


class BertFinetunedRACE(BaseSingleChoiceMRC):
    Tokenizer = BertTokenizer
    Model = BertModel

    def __init__(self,
                 pretrained_model_name_or_path,
                 device="cpu",
                 dropout_rate=.1,
                 max_length=512,
                 ):
        super(BertFinetunedRACE, self).__init__(pretrained_model_name_or_path,
                                                device=device,
                                                dropout_rate=dropout_rate,
                                                max_length=max_length
                                                )


class AlbertFinetunedRACE(BaseSingleChoiceMRC):
    Tokenizer = AlbertTokenizer
    Model = AlbertModel

    def __init__(self,
                 pretrained_model_name_or_path,
                 device="cpu",
                 dropout_rate=.1,
                 max_length=512,
                 ):
        super(AlbertFinetunedRACE, self).__init__(pretrained_model_name_or_path,
                                                  device=device,
                                                  dropout_rate=dropout_rate,
                                                  max_length=max_length,
                                                  )


class RobertaFinetunedRACE(BaseSingleChoiceMRC):
    Tokenizer = AlbertTokenizer
    Model = AlbertModel

    def __init__(self,
                 pretrained_model_name_or_path,
                 device="cpu",
                 dropout_rate=.1,
                 max_length=512,
                 ):
        super(RobertaFinetunedRACE, self).__init__(pretrained_model_name_or_path,
                                                   device=device,
                                                   dropout_rate=dropout_rate,
                                                   max_length=max_length,
                                                   )
