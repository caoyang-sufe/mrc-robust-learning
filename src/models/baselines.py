# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

from torch.nn import Module, Linear, Dropout
from transformers import AutoTokenizer, AutoModel


class Baseline(Module):
    model_name = 'baseline'
    Tokenizer = AutoTokenizer
    Model = AutoModel

    def __init__(self,
                 pretrained_model_name_or_path,
                 device='cpu',
                 hidden_size=1024,
                 dropout_rate=.1,

                 ):
        super(Baseline, self).__init__()
        self.tokenizer = Tokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = Model.from_pretrained(pretrained_model_name_or_path).to(device)
        self.dropout = Dropout(p=dropout_rate, inplace=False)
        self.classifier = Linear(in_features=1024, out_features=1, bias=True)

    def forward(self, article, question, option):
        inputs = tokenizer(article, question, option, return_tensors='pt')
        y = self.classifier(self.dropout(model(**inputs).pooler_output))
        return res


class BERTLargeFinetunedRACE(Baseline):

    pass