import pdb
import random
"""
处理句子数据，通用版本
"""

def build_vocab(data, save_path="vocab", save="True"):
    """
    从数据中建立词汇表list
    :param data: 二维列表，表示第i行的第j个token
    :save_path: 要保存/读取词典文件的地址
    :save : 是否保存，保存则为True
    :return: 词表list
    """
    import os
    vocab = []
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as file:
            for word in file.readlines():
                vocab.append(word[:-1])  # 去掉换行
    else:
        word_set = set()
        for line in data:
            for word in line:
                if word not in word_set: word_set.add(word)
        vocab = list(word_set)
        if save:
            with open(save_path, 'w', encoding='utf-8') as file:
                for word in vocab:
                    file.write(word + "\n")
    return vocab

def tokenize(data, tokenizer, still=True, mode="dict", end_pad=False):
    """
    将二维列表数据tokenize
    :param still 原地改变
    :param mode tokenizer的形式，这里提供了
          字典
          bert
    :param end_pad 适应bert，首尾两端添加0
    """
    result = data if still else [len(i) * [0] for i in data]
    if mode == "dict":
        for i in range(len(data)):
            for j in range(len(data[i])):
                result[i][j] = tokenizer[data[i][j]]
            if end_pad: result[i] = [0] + result[i] + [0]
    elif mode == "bert":
        SEP = tokenizer.convert_tokens_to_ids("[SEP]")
        CLS = tokenizer.convert_tokens_to_ids("[CLS]")
        for i in range(len(data)):
            for j in range(len(data[i])):
                result[i][j] = tokenizer.convert_tokens_to_ids(data[i][j])
            result[i] = [CLS] + result[i] + [SEP]
    else:
        raise ("invalid mode!")

    return result

def list2dict(l):
    """
    将列表转化为字典，即序号与内容倒过来
    """
    result = dict()
    for i in range(len(l)):
        result[l[i]] = i
    return result

def get_len(data):
    """
    获取数据中每行的长度
    :param data:二维列表
    """
    return [len(i) for i in data]

def padding(data, labels=None, max_len=0):
    """
    将data和labels(有的话)以最大长度填0（原地填）
    :param data: 二维列表数据
    :param labels:  顺带着data的label
    :param max_len: 给定padding的最大长度，默认没有，即为0
    :return: 原地变换   padding后的data和labels,
    """
    if max_len == 0: max_len = max([len(i) for i in data])
    if labels is None:
        for i in range(len(data)):
            data[i] = data[i] + [0] * (max_len - len(data[i]))
    else:
        for i in range(len(data)):
            data[i] = data[i] + [0] * (max_len - len(data[i]))
            labels[i] = labels[i] + [0] * (max_len - len(labels[i]))

def cut_sentence(data, labels, max_length):
    """
    将过于长的句子剪掉
    :param data: 二维列表
    :param labels: 二维列表
    :param max_length: 允许最大句长
    :return: 原地改变
    """
    for i in range(len(data)):
        if len(data[i]) > max_length:
            data[i] = data[i][:max_length]
            if labels: labels[i] = labels[i][:max_length]

def cut_data(data,rate="default",shuffle_seed=False):
    """
    切分列表（数据集常用）
    :param data:列表
    :param rate:切分比率，默认8:1:1 (不是严格的比率，会有整数问题)
    :param dev:是否需要dev
    :return:
    """
    if shuffle_seed:
        random.seed(shuffle_seed)
        random.shuffle(data)
    if rate=="default":rate=[0.8,0.1,0,1]
    elif sum(rate)!=1: raise Exception("概率和不等于1")
    indexs=[0]
    for i in range(0,len(rate)-2):
        index=indexs[-1]+int(len(data)*rate[i])
        print(index)
        indexs.append(index)
    cutted_data=tuple()
    for i in range(len(indexs)-1):
        cutted_data+=(data[indexs[i]:indexs[i+1]],)
    cutted_data+=(data[indexs[-1]:],)
    return cutted_data
def cut_data_ave(data,ave_len=4):
    """
    根据长度平分
    返回的data比原来多一维
    """
    new_data=[]
    index=0
    while index+ave_len<len(data):
        new_data.append(data[index:index+ave_len])
        index+=ave_len
    new_data.append(data[index:])
    return new_data


def add_embedding_file(files):
    """
    word_embedding
    """


    def add(self, instance):
        '''
        添加实体
        '''
        if instance not in self.instance2id:
            if instance not in self.alphabet:
                #构建词典
                self.alphabet[instance] = 1
            else:
                self.alphabet[instance] += 1
            if self.alphabet[instance] >= self.freq:   # 出现一定的频率才能写入词典中
                #统计词频
                self.instance2id[instance] = len(self.instances)
                self.instances.append(instance)
                self.begin_index += 1

    # print("Loading pretrained embedding from %s" % file)
    if files is None:
        return
    embed_dict = dict()
    if files == None:
        return
    with open(files, 'r', encoding='utf-8') as fin:
        fin.readline()
        for line in fin:
            line = line.rstrip().split()
            word = line[0]
            embed_dict[word] = ' '.join(line[1:])
    for word in embed_dict.keys():
        add(word)

def del_no_word_sentences(sentences,vocab_dict,still=True):
    """
    删除不在词典中的句子
    先记录，再删除
    :param vocab_dict 词典字典
    :param still:原地删除
    """
    keys=set(vocab_dict.keys())
    no_word_sentences_idx=[]
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            if sentences[i][j] not in keys:
                no_word_sentences_idx.append(i)
                break
    ite=0
    for i in no_word_sentences_idx:
        del sentences[i-ite]
        ite+=1
    if not still:
        new_sentences=[[0]*len(sentences[i]) for i in sentences]
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                new_sentences[i][j] = sentences[i][j]
        return new_sentences

def del_no_word_sentences(sentences,tags,vocab_dict,still=True):
    """
    删除不在词典中词所在的句子,同时删除对应标签的句子
    先记录，再删除
    :param vocab_dict 词典字典
    :param still:原地删除
    """
    keys=set(vocab_dict.keys())
    no_word_sentences_idx=[]
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            if sentences[i][j] not in keys:
                no_word_sentences_idx.append(i)
                break
    ite=0
    for i in no_word_sentences_idx:
        del sentences[i-ite]
        del tags[i-ite]
        ite+=1
    if not still:
        new_sentences=[[0]*len(sentences[i]) for i in sentences]
        new_tags=[[0]*len(tags[i]) for i in tags]
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                new_sentences[i][j] = sentences[i][j]
                new_tags[i][j] = tags[i][j]
        return new_sentences,new_tags

def merge_list(l:list)->list:
    new_l=[]
    for i in l:new_l+=i
    return new_l