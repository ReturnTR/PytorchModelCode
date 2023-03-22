# 基本的工具
# 包含内置类型的一些常用操作
# 该文件要求不能导入任何的包
# 如果是对特定类型进行操作，格式为（操作_类型）

def sort_dict(source_dict,use_key=False,reverse=True):
    """
    对字典排序，可按值或按键
    :param source_dict:原字典
    :param ues_key:使用字典的键作为排序的键
    :param reverse: True默认从大到小
    :return:排序后的字典
    """
    if use_key:
        return {k: v for k, v in sorted(source_dict.items(), key=lambda item: item[0], reverse=reverse)}
    else:
        return {k: v for k, v in sorted(source_dict.items(), key=lambda item: item[1], reverse=reverse)}

def count_list(l,reverse=True):
    """
    将列表出现的相同字符进行计数
    :param l:列表
    :return:字典{值:出现次数}
    """
    res=dict()
    for i in l:res[i]=res.setdefault(i,0)+1
    sort_dict(res,reverse=reverse)
    return res

def avg_list(l):
    """list平均值"""
    return sum(l)/len(l)

def percentage(num):
    """将小数转化成百分比格式,保留两位小数"""
    return str(int(num*10000)/100)+"%"

def cut_data(data,rate="default"):
    """
    切分列表（数据集常用）
    :param data:列表
    :param rate:切分比率，默认8:1:1 (不是严格的比率，会有整数问题)
    :param dev:是否需要dev
    :return:
    """
    if rate=="default":rate=[0.8,0.1,0.1]
    elif sum(rate)!=1: raise Exception("概率和不等于1")
    amount=[int(len(data)*i) for i in rate]
    for i in range(1,len(amount)):
        amount[i]+=amount[i-1]
    amount[-1]=len(data)
    amount=[0]+amount
    cutted_data=tuple()
    for i in range(len(amount)-1):
        cutted_data+=(data[amount[i]:amount[i+1]],)

    return cutted_data

def count_dict_list(data):
    """
    记录字典列表的数量信息
    :param data:
    :return:
    """
    res=dict()
    res["key数量"]=len(data)
    res["value数量"]=sum([len(i) for i in data.values()])
    res["平均key占比"]=res["value数量"]/res["key数量"]
    res["只有一条的key数量"]=sum([1 for i in data.values() if len(i)==1])
    res["只有一条的key占比"]=int(res["只有一条的key数量"]/res["key数量"]*10000)/10000
    return res

def print_dict(data):
    """将字典以换行形式输出"""
    for key,value in data.items():
        print(key,":",value)

class DictCount():

    def __init__(self):
        self.infobox=dict()
    def add(self,item):

        if item in self.infobox:
            self.infobox[item]+=1
        else:
            self.infobox[item]=1
    def get(self):
        self.infobox=sort_dict(self.infobox)
        return self.infobox

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

def cut_sentence(data, labels=None, max_length=200):
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
