from copy import deepcopy
from tqdm import tqdm

class Tokenizer:
    """
    两种数据类型的映射的抽象类

    """
    def tokenize_A2B(self,data):pass
    def tokenize_B2A(self,data):pass

class DictTokenizer(Tokenizer):

    def __init__(self,data2label):
        """
        输入字典构建词典
        如果是一对一的话构建反向词典
        """
        self.data2label=data2label
        label2data=dict()
        for k,v in data2label.items():
            if k in label2data: return
            else:label2data[v]=k
        self.label2data=label2data

    def tokenize(self,data,data2label):
        result = deepcopy(data)
        if isinstance(data,list):
            if isinstance(data[0],list):
                # 二维列表
                for i in range(len(data)):
                    for j in range(len(data[i])):
                        result[i][j] = data2label[data[i][j]]
            else:
                # 一维列表
                for i in range(len(data)):
                    result[i]=data2label[data[i]]
        else:
            # 标量
            result=data2label[data]
        return result

    def tokenize_A2B(self,data):
        return self.tokenize(data,self.data2label)

    def tokenize_B2A(self,labels):
        return self.tokenize(labels,self.label2data)


class bertTokenizer(Tokenizer):

    # 原地改变

    def __init__(self,bert_path):
        from transformers import BertTokenizer
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        
    def tokenize_A2B(self,data,add_sep=True):
        # 首末添加特殊字符
        # 原地改变
        SEP = self.tokenizer.convert_tokens_to_ids("[SEP]")
        CLS = self.tokenizer.convert_tokens_to_ids("[CLS]")
        for i in tqdm(range(len(data))):
            data[i] = self.tokenizer.convert_tokens_to_ids(data[i])
            if add_sep:
                data[i]=[CLS] + data[i] + [SEP]

    def tokenize_B2A(self,data,remove_sep=False):
        for i in range(len(data)):
            data[i] = self.tokenizer.decode(data[i])
            if remove_sep:
                if "[SEP]" in data[i]:data[i].remove("[SEP]")
                if "[CLS]" in data[i]: data[i].remove("[CLS]")

