# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

import re
import torch
from torch.nn import Module, Linear, Dropout, CrossEntropyLoss
from transformers import AutoTokenizer, AutoModel, AlbertTokenizer, AlbertModel, BertTokenizer, BertModel


class Baseline(Module):
    Tokenizer = AutoTokenizer
    Model = AutoModel
    criterion = CrossEntropyLoss()
    model_name = "baseline"
    alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    alphabet2id = {alphabet: i for i, alphabet in enumerate(alphabets)}
    underline_regex = re.compile("_+")

    def __init__(self,
                 pretrained_model_name_or_path,
                 device="cpu",
                 dropout_rate=.1,
                 max_length=512,
                 ):
        super(Baseline, self).__init__()
        self.device = device
        self.max_length = max_length
        self.tokenizer = self.Tokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = self.Model.from_pretrained(pretrained_model_name_or_path).to(device)
        self.hidden_size = self.model.config.hidden_size
        self.dropout = Dropout(p=dropout_rate, inplace=False)
        self.classifier = Linear(in_features=self.hidden_size, out_features=1, bias=True)

    def forward(self, batch_data, mode="train"):
        batch_size = len(batch_data)
        inputs, labels = self.preprocess(batch_data)
        outputs = self.classifier(self.dropout(self.model(**inputs).pooler_output)).reshape(batch_size, -1)   # (batch_size, n_option)
        print(outputs)
        if mode == "train":
            return self.criterion(outputs, labels)
        elif mode in ["dev", "test"]:
            predicts = torch.argmax(outputs, dim=-1)
            print(predicts)
            return sum(predicts == labels) / batch_size

    def preprocess(self, batch_data):
        input_ids = list()
        attention_mask = list()
        labels = list()
        for data in batch_data:
            article = data["article"]
            question = data["question"]
            options = data["options"]
            answer = data["answer"]
            labels.append(self.alphabet2id[answer])
            flag = len(self.underline_regex.findall(question)) == 1
            for option in options:
                question_option = self.underline_regex.sub(option, question) if flag else question + ' ' + option
                tokenized_input = self.tokenizer(article,
                                                 question_option,
                                                 add_special_tokens=True,
                                                 max_length=self.max_length,
                                                 padding="max_length",
                                                 truncation=True,
                                                 return_overflowing_tokens=False,
                                                 )
                input_ids.append(tokenized_input["input_ids"])
                attention_mask.append(tokenized_input["attention_mask"])
        input_ids = torch.LongTensor(input_ids).to(self.device)             # (batch_size × n_option, max_length)
        attention_mask = torch.LongTensor(attention_mask).to(self.device)   # (batch_size × n_option, max_length)
        labels = torch.LongTensor(labels).to(self.device)                   # (batch_size, )
        return {"input_ids": input_ids, "attention_mask": attention_mask}, labels


class BERTLargeFinetunedRACE(Baseline):
    Tokenizer = BertTokenizer
    Model = BertModel
    model_name = "bert-large-finetuned-race"

    def __init__(self,
                 pretrained_model_name_or_path,
                 device="cpu",
                 dropout_rate=.1,
                 max_length=512,
                 ):
        super(BERTLargeFinetunedRACE, self).__init__(pretrained_model_name_or_path,
                                                     device=device,
                                                     dropout_rate=dropout_rate,
                                                     max_length=max_length
                                                     )


class ALBERTLargeFinetunedRACE(Baseline):
    Tokenizer = AlbertTokenizer
    Model = AlbertModel
    model_name = "albert-large-finetuned-race"

    def __init__(self,
                 pretrained_model_name_or_path,
                 device="cpu",
                 dropout_rate=.1,
                 max_length=512,
                 ):
        super(ALBERTLargeFinetunedRACE, self).__init__(pretrained_model_name_or_path,
                                                       device=device,
                                                       dropout_rate=dropout_rate,
                                                       max_length=max_length,
                                                       )
