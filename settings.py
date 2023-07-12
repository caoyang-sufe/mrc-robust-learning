# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

import os

DATA_DIR = r"D:\resource\data"
# DATA_DIR = "/nfsshare/home/caoyang/code/data"
DATA_SUMMARY = {"RACE": {"path": os.path.join(DATA_DIR, "RACE")},
                "DREAM": {"path": os.path.join(DATA_DIR, "DREAM")},
                "COPA": {"path": os.path.join(DATA_DIR, "BCOPA-CE")},
                }

MODEL_DIR = r"D:\resource\model\huggingface\common"
# MODEL_DIR = "/nfsshare/home/caoyang/code/model"
MODEL_SUMMARY = {"albert-base-v1": {"path": os.path.join(MODEL_DIR, "albert-base-v1")},
                 "albert-large-v1": {"path": os.path.join(MODEL_DIR, "albert-large-v1")},
                 "albert-base-v2": {"path": os.path.join(MODEL_DIR, "albert-base-v2")},
                 "albert-large-v2": {"path": os.path.join(MODEL_DIR, "albert-large-v2")},
                 "bert-base-uncased": {"path": os.path.join(MODEL_DIR, "bert-base-uncased")},
                 "bert-large-uncased": {"path": os.path.join(MODEL_DIR, "bert-large-uncased")},
                 "roberta-base": {"path": os.path.join(MODEL_DIR, "roberta-base")},
                 "roberta-large": {"path": os.path.join(MODEL_DIR, "roberta-large")},
                 }

CKPT_DIR = "ckpt"
LOG_DIR = "log"
