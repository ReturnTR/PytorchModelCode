from torch.utils.data import Dataset
from copy import deepcopy
import torch
from ..CommonTools.BasicTool import padding
from ..CommonTools.OSTool import get_file_of_prefix
from ..CommonTools.JsonTool import *
import pdb

# 不能将最顶层的引入

# 允许缓存


class StandardDataset(Dataset):
    """
    提供dataset标准
    """
    def __init__(self,data_path,load_file_fn):
        """
        加载数据，包括data和label
        :param data_path: 输入文件夹，里面必须含有train，test
        :param load_file_fn: 文件的打开方法，获取数据，数据的格式为data列表，label列表

        返回的数据：
            self.train_data
            self.train_labels
            self.test_data
            self.test_labels
            self.dev_data
            self.dev_labels
        """
        print("opening data...")
        train_path = data_path + "/train"
        test_path = data_path + "/test"
        dev_path = data_path + "/dev"

        train_path=get_file_of_prefix(train_path)
        test_path=get_file_of_prefix(test_path)
        dev_path = get_file_of_prefix(dev_path)

        import os
        if not os.path.exists(dev_path):
            dev_path = test_path

        # 获取源数据
        train_data, train_labels = load_file_fn(train_path)


        # 若只有train则将数据集进行切分
        if not os.path.exists(test_path):
            train_data, train_labels, test_data, test_labels, dev_data, dev_labels = cut_data(train_data, train_labels)
        else:
            test_data, test_labels = load_file_fn(test_path)
            dev_data, dev_labels = load_file_fn(dev_path)

        

        # 保存
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.dev_data = dev_data
        self.dev_labels = dev_labels

        # 获取句长
        self.train_len = [len(i) for i in train_data]
        self.test_len = [len(i) for i in test_data]
        self.dev_len = [len(i) for i in dev_data]


        

    def tokenize(self,data_tokenizer=None,label_tokenizer=None):
        # 如有必要，对data或label进行文字到数字的转换

        # 保存原始数据
        self.train_data_without_tokenize = deepcopy(self.train_data)
        self.train_labels_without_tokenize = deepcopy(self.train_labels)
        self.test_data_without_tokenize = deepcopy(self.test_data)
        self.test_labels_without_tokenize = deepcopy(self.test_labels)
        self.dev_data_without_tokenize = deepcopy(self.dev_data)
        self.dev_labels_without_tokenize = deepcopy(self.dev_labels)
        print("tokenizing...")
        if data_tokenizer!=None:
            data_tokenizer.tokenize_A2B(self.train_data)
            data_tokenizer.tokenize_A2B(self.test_data)
            data_tokenizer.tokenize_A2B(self.dev_data)

        if label_tokenizer!=None:
            self.train_labels = data_tokenizer.tokenize_A2B(self.train_labels)
            self.test_labels = data_tokenizer.tokenize_A2B(self.test_labels)
            self.dev_labels = data_tokenizer.tokenize_A2B(self.dev_labels)

    def padding(self,pad_label=True):
        # 尾部填充
        padding(self.train_data)
        padding(self.test_data)
        padding(self.dev_data)
        if pad_label:
            padding(self.train_labels)
            padding(self.test_labels)
            padding(self.dev_labels)

    def to_tensor(self):
        # 转换成tensor
        self.train_data = torch.as_tensor(self.train_data).long()
        self.train_labels = torch.as_tensor(self.train_labels)
        self.test_data = torch.as_tensor(self.test_data).long()
        self.test_labels = torch.as_tensor(self.test_labels)
        self.dev_data = torch.as_tensor(self.dev_data).long()
        self.dev_labels = torch.as_tensor(self.dev_labels)

    def set_mode(self, mode):
        self.mode = mode

    def get_no_tokenize_data(self):
        if self.mode == "train":
            return self.train_data_without_tokenize, self.train_labels_without_tokenize
        elif self.mode == "test":
            return self.test_data_without_tokenize, self.test_labels_without_tokenize
        elif self.mode == "dev":
            return self.dev_data_without_tokenize, self.dev_labels_without_tokenize
        else:
            raise ("Wrong mode!")

    def __getitem__(self, item):
        if self.mode == "train":
            return self.train_data[item], self.train_labels[item], self.train_len[item]
        elif self.mode == "test":
            return self.test_data[item], self.test_labels[item], self.test_len[item]
        elif self.mode == "dev":
            return self.dev_data[item], self.dev_labels[item], self.dev_len[item]
        else:
            raise ("Wrong mode!")

    def __len__(self):
        if self.mode == "train":
            return len(self.train_data)
        elif self.mode == "test":
            return len(self.test_data)
        elif self.mode == "dev":
            return len(self.dev_data)
        else:
            raise ("Wrong mode!")


    def save_cache(self,data_path):
        save_json([self.train_data,self.train_labels,self.train_len,self.test_data,self.test_labels,self.test_len,self.dev_data,self.dev_labels,self.dev_len],data_path+"/cache.json")


    def load_cache(self,data_path):
        cache=load_json(data_path+"/cache.json")
        self.train_data,self.train_labels,self.train_len,self.test_data,self.test_labels,self.test_len,self.dev_data,self.dev_labels,self.dev_len=cache[0],cache[1],cache[2],cache[3],cache[4],cache[5],cache[6],cache[7],cache[8]