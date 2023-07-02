# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

import os

DATA_ROOT = r"D:\resource\data"
# DATA_ROOT = "/nfsshare/home/caoyang/code/data"
DATA_PATH = {"RACE": os.path.join(DATA_ROOT, "RACE"),
             "DREAM": os.path.join(DATA_ROOT, "DREAM"),
             "COPA": os.path.join(DATA_ROOT, "BCOPA-CE"),
             }

MODEL_ROOT = r"D:\resource\model\huggingface\common"
# MODEL_ROOT = "/nfsshare/home/caoyang/code/model"
PLM_PATH = {"albert-base-v1": os.path.join(MODEL_ROOT, "albert-base-v1"),
            "albert-large-v1": os.path.join(MODEL_ROOT, "albert-large-v1"),
            "albert-base-v2": os.path.join(MODEL_ROOT, "albert-base-v2"),
            "albert-large-v2": os.path.join(MODEL_ROOT, "albert-large-v2"),
            "bert-base-uncased": os.path.join(MODEL_ROOT, "bert-base-uncased"),
            "bert-large-uncased": os.path.join(MODEL_ROOT, "bert-large-uncased"),
            "roberta-base": os.path.join(MODEL_ROOT, "roberta-base"),
            "roberta-large": os.path.join(MODEL_ROOT, "roberta-large"),
            }
