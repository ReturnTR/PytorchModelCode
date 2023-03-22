import torch
import torch.nn as nn
from .bert_model import Bert_Encoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence,pad_sequence
from torchcrf import CRF
import pdb



class BERT_CRF(nn.Module):
    def __init__(self,
        bert_path,
        config
    ):
        super().__init__()
        self.bert_embedding=Bert_Encoder(bert_path, config.embedding_dim)
        self.hidden2tag=nn.Linear(config.embedding_dim,config.tag_num)
        self.crf=CRF(config.tag_num,batch_first=True)
        self.loss_calculator = nn.CrossEntropyLoss(ignore_index=0) #表示0不在计算范围内

    def forward(self,x,y):
        """
        :param x :batch_size*max_sentence_len
        """
        bert_out = self.bert_embedding(x)
        output=self.hidden2tag(bert_out)
        if self.training:
            mask=y.gt(0)
            output=output[:,1:]
            y=y[:,1:]
            mask=mask[:,1:]
            loss= -self.crf(output,y,mask=mask)
            # output = output.contiguous().view(-1, output.shape[-1]) #将维度变成二维
            # real_tagseq = y.contiguous().view(-1) #变成一维
            # loss = self.loss_calculator(output, real_tagseq)
            return loss
        else:
            mask=y.gt(0)
            # mask第一位不能为0，否则会出错
            output=output[:,1:]
            mask=mask[:,1:]
            predict_tag=self.crf.decode(output,mask=mask)
            for i in range(len(predict_tag)):
                predict_tag[i]=[0]+predict_tag[i]+[0]
            predict_tag=[torch.as_tensor(i,device=x.device) for i in predict_tag]
            predict_tag=pad_sequence(predict_tag,batch_first=True)
            # predict_tag = torch.argmax(output, 2)

            return predict_tag

class BERT_CRF_Config():
    def __init__(self,
        tag_num=0,
        device="cpu",
        embedding_dim=768,
        learning_rate=10e-5
    
    ):
        self.tag_num=tag_num
        self.device=device
        self.embedding_dim=embedding_dim
        self.learning_rate=learning_rate