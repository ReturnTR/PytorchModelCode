import torch
import torch.nn as nn
from transformers import BertModel
import pdb

class Bert_Encoder(nn.Module):
    def __init__(self, bert_path, bert_dim):
        super(Bert_Encoder, self).__init__()
        self.bert_dim = bert_dim
        self.bert = BertModel.from_pretrained(bert_path)                                                                                                                                                                   

    def forward(self, subword_idxs):
        mask = subword_idxs.gt(0)
        bert_outs= self.bert(
            subword_idxs,
            token_type_ids=None,
            attention_mask=mask
        )     
        return bert_outs[0] # last_hidden_state 最后一层的隐藏状态


class FineTuneTagger(nn.Module):
    def __init__(self, bert_path, tag_size,hidden_dim=768):
        super(FineTuneTagger, self).__init__()
        self.bert_encoder = Bert_Encoder(bert_path, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        """
        nn.CrossEntropyLoss:
            Input: N*C
            Target: N
            output = loss(Input, Target)
        """
        self.loss_calculator = nn.CrossEntropyLoss(ignore_index=0) #表示0不在计算范围内

    def forward(self, subword_idxs,real_tagseq):
        bert_outs = self.bert_encoder(subword_idxs)
        out = self.hidden2tag(bert_outs) #batch_size*sentence_len*tag_num
        if self.training:
            out = out.contiguous().view(-1, out.shape[-1]) #将维度变成二维
            real_tagseq = real_tagseq.contiguous().view(-1) #变成一维
            loss = self.loss_calculator(out, real_tagseq)
            return loss
        else:
            predict_tag = torch.argmax(out, 2)
            return predict_tag


class FineTuneClassifier(nn.Module):
    def __init__(self,  bert_path ,tag_size,output_label_num,hidden_dim=768):
        super(FineTuneClassifier, self).__init__()
        self.bert_encoder = Bert_Encoder(bert_path, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        self.output_label_num=output_label_num
        """
        nn.CrossEntropyLoss:
            Input: N*C
            Target: N
            output = loss(Input, Target)
        """
        self.loss_calculator = nn.CrossEntropyLoss() #表示0不在计算范围内

    def forward(self, subword_idxs,real_tagseq):
        bert_outs = self.bert_encoder(subword_idxs)
        out = self.hidden2tag(bert_outs) #batch_size*sentence_len*tag_num
        out=out[:,0,:]   # 获取CLS的结果
        if self.training:
            # out = out.contiguous().view(-1)
            # real_tagseq = real_tagseq.contiguous().view(-1) #变成一维
            loss = self.loss_calculator(out, real_tagseq)
            return loss
        else:
            # out=torch.nn.functional.softmax(out.float(),dim=1)
            # 指定输出哪一维度的值
            if self.output_label_num!=None:
                return out[:,self.output_label_num]
            max_predict_value=torch.max(out,1) # 包含indices和value
            print(max_predict_value)
            pdb.set_trace()
            predict_tag = torch.argmax(out, 1)
            return max_predict_value


            