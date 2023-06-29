# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

import re
import torch
from torch.nn import Module, Linear, Dropout
from transformers import AutoTokenizer, AutoModel


class Baseline(Module):
    Tokenizer = AutoTokenizer
    Model = AutoModel
    model_name = "baseline"
    alphabets = ['A', 'B', 'C', 'D']
    alphabet2id = {alphabet: i for i, alphabet in enumerate(alphabets)}
    underline_regex = re.compile("_+")

    def __init__(self,
                 pretrained_model_name_or_path,
                 device="cpu",
                 hidden_size=1024,
                 dropout_rate=.1,
                 max_length=512,
                 ):
        super(Baseline, self).__init__()
        self.max_length = max_length
        self.device = device
        self.tokenizer = Tokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = Model.from_pretrained(pretrained_model_name_or_path).to(device)
        self.dropout = Dropout(p=dropout_rate, inplace=False)
        self.classifier = Linear(in_features=hidden_size, out_features=1, bias=True)

    def forward(self, batch_data):
        inputs_list = list()
        for article_id, question_id, article, question, options, answer in batch_data:
            flag = question.find('_') == -1
            inputs = list()
            for option in options:
                question_choice = question + ' ' + option if flag else question.replace('_', option)
                input_ = self.tokenizer(article,
                                        question_choice,
                                        add_special_tokens=True,
                                        max_length=self.max_length,
                                        padding="max_length",
                                        truncation=True,
                                        return_overflowing_tokens=False,
                                        )
                inputs.append(input_)
            inputs_list.append(inputs)
            labels.append(option2id[answer])
        input_ids = torch.LongTensor([[x["input_ids"] for x in inputs] for inputs in inputs_list])
        attention_mask = torch.LongTensor(
            [[x["attention_mask"] for x in inputs] for inputs in inputs_list]) if "attention_mask" in inputs_list[0][
            0] else None
        del inputs
        model_inputs = {'input_ids': input_ids.to(device), 'attention_mask': attention_mask.to(device)}
        logits = model(**model_inputs).logits
        inputs = tokenizer(article, question, option, return_tensors='pt')
        y = self.classifier(self.dropout(model(**inputs).pooler_output))
        return res

    def prepare_inputs(self, batch_data):
        batch_encoded_inputs = list()
        batch_labels = list()
        for article_id, question_id, article, question, options, answer in batch_data:
            flag = len(underline_regex.findall(question)) == 1
            encoded_inputs = list()
            for option in options:
                question_option = underline_regex.sub(option, question) if flag else question + ' ' + option
                encoded_input = self.tokenizer(article,
                                               question_option,
                                               add_special_tokens=True,
                                               max_length=self.max_length,
                                               padding="max_length",
                                               truncation=True,
                                               return_overflowing_tokens=False,
                                               )
                encoded_inputs.append(encoded_input)
            batch_encoded_inputs.append(encoded_inputs)
            batch_labels.append(self.alphabet2id[answer])
        input_ids = [[x["input_ids"] for x in encoded_inputs] for encoded_inputs in batch_encoded_inputs]
        input_ids = torch.LongTensor(input_ids).to(self.device)
        attention_mask = [[x["attention_mask"] for x in inputs] for inputs in inputs_list]
        attention_mask = torch.LongTensor(attention_mask).to(self.device)
        del encoded_input, encoded_inputs, batch_encoded_inputs
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class BERTLargeFinetunedRACE(Baseline):
    model_name = "bert-large-finetuned-race"

    def __init__(self,
                 pretrained_model_name_or_path,
                 device="cpu",
                 hidden_size=1024,
                 dropout_rate=.1,
                 ):
        super(BERTLargeFinetunedRACE, self).__init__(pretrained_model_name_or_path,
                                                     device=device,
                                                     hidden_size=hidden_size,
                                                     dropout_rate=dropout_rate,
                                                     )

    def forward(self):
        pass
