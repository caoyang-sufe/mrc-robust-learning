# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

import re
from src.models.baselines import BertFinetunedRACE, AlbertFinetunedRACE
from src.tools.data_tools import yield_race_sample
from transformers import AlbertTokenizer, AlbertModel

model_path = r"D:\resource\model\huggingface\common\albert-base-v1"
race_path = r"D:\resource\data\RACE"

# model = ALBERTLargeFinetunedRACE(pretrained_model_name_or_path=model_path, device="cpu")
# model.eval()
generator = yield_race_sample(race_path, ["train"], ["high", "middle"], batch_size=2)
underline_regex = re.compile("_+", re.U)
for i, batch_data in enumerate(generator):
    print(i)
    for data in batch_data:
        article = data["article"]
        question = data["question"]
        options = data["options"]
        answer = data["answer"]
        flag = len(underline_regex.findall(question)) == 1
        for option in options:
            try:
                question_option = underline_regex.sub(option.replace('\\', ' '), question) if flag else question + ' ' + option
            except Exception as e:
                print(e)
                print(question, option)
                print(question_option)
