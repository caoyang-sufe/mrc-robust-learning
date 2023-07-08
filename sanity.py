# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

from src.models.baselines import BertLargeFinetunedRACE, AlbertLargeFinetunedRACE
from src.tools.data_tools import yield_race_sample
from transformers import AlbertTokenizer, AlbertModel

model_path = r"D:\resource\model\huggingface\common\albert-base-v1"
race_path = r"D:\resource\data\RACE"

model = ALBERTLargeFinetunedRACE(pretrained_model_name_or_path=model_path, device="cpu")
model.eval()
generator = yield_race_sample(race_path, ["train"], ["high", "middle"], batch_size=2)

for batch_data in generator:
    a = model(batch_data, mode="train")
    b = model(batch_data, mode="dev")

    print(batch_data[0]["answer"])
    print(batch_data[1]["answer"])

    print(a)
    print(b)
    break