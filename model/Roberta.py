import torch
import torch.nn as nn
from transformers import BertModel


def tset(bert_path):

    bert = BertModel.from_pretrained(bert_path)
    print(bert)