

from torch.utils.data import Dataset,DataLoader
import torch
from tqdm import tqdm
from .Standard import StandardDataset
from ..CommonTools.JsonTool import load_json
from ..CommonTools.BasicTool import cut_sentence
import pdb
import os


# 句子分类的数据集


def load_sentence_file(filename):
    """如果有[SEP]的话需要将其分开"""
    data=load_json(filename)
    for i in range(len(data)):
        if "[SEP]" in data[i][0]:
            temp=data[i][0].split("[SEP]")
            data[i][0]=[]
            for j in temp:
                data[i][0]+=list(j)
                data[i][0]+=["[SEP]"]
            data[i][0]=data[i][0][:-1] # 去掉最后一个[SEP]
        else:
            data[i][0]=list(data[i][0])
    return [i[0] for i in data],[i[1] for i in data]


class SCDataset(StandardDataset):

    def __init__(self, data_path, data_tokenizer) -> None:

        if os.path.exists(data_path+"/cache.json"):
            self.load_cache()
            return 

        super().__init__(data_path, load_sentence_file)

        cut_sentence(self.train_data)
        cut_sentence(self.test_data)
        cut_sentence(self.dev_data)


        self.tokenize(data_tokenizer=data_tokenizer)


        self.padding(pad_label=False)

        # self.to_tensor()

        print("dataset building over!")

        self.save_cache()

def SCevaluate(model,dataset,tag2num,config):


    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )
    model.eval()


    y_all=[]
    y_hat_indices_all=[]

    for x, y, batch_len in tqdm(dataloader):
        x= x.to(config.device)
        y =y.to(config.device)
        x=x.T[:max(batch_len)].T
        batch_len,indices=batch_len.sort(descending=True)
        y=y[indices]
        x=x[indices]
        y_hat = model(x,y)
        y_hat_values=y_hat.values
        y_hat_indices=y_hat.indices
        y_all+=list(y)
        y_hat_indices_all+=list(y_hat_indices)
    
    print(len(y_all))

    print(len(y_hat_indices_all))

    right_count=sum([1 for i in range(len(y_all)) if y_all[i]==y_hat_indices_all[i]])

    print("正确率：",right_count/len(y_all))



    return 


def SClabel_x(x,model,dataset,config):
    """
    x: sentenece_num*sentence_len
    """
    model.eval()
    
    import copy
    y_hat_all=[]

    # 以200为间隔
    if len(x)>200:
        x_big=cut_data_ave(x)
        for x in x_big:
            x_words= copy.deepcopy(x)
            x=[list(i) for i in x]
            tokenize(x, tokenizer=dataset.word_tokenizer, mode="bert")
            padding(x)
            x = torch.as_tensor(x).long()
            x=x.to(config.device)
            y_hat = model(x,None)

            y_hat=[[x_words[i],dataset.num2tag[y_hat.indices[i]],float(y_hat.values[i])] for i in range(len(y_hat.indices))]

            y_hat_all+=y_hat

    # y_hat_all = sorted(y_hat_all, key=lambda d: d[2], reverse=True)

    return y_hat_all


def SClabel(model,dataset,tag2num,config):


    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )
    model.eval()


    y_all=[]
    y_hat_indices_all=[]

    y_hat_values_all=[]
    for x, y, batch_len in tqdm(dataloader):
        x= x.to(config.device)
        y =y.to(config.device)
        x=x.T[:max(batch_len)].T
        batch_len,indices=batch_len.sort(descending=True)
        y=y[indices]
        x=x[indices]
        y_hat = model(x,y)
        y_hat_values=y_hat.values
        y_hat_indices=y_hat.indices
        y_all+=list(y)
        y_hat_indices_all+=list(y_hat_indices)
        y_hat_values_all+=list(y_hat_values)
    
    print(len(y_all))

    print(len(y_hat_indices_all))

    right_count=sum([1 for i in range(len(y_all)) if y_all[i]==y_hat_indices_all[i]])

    print("正确率：",right_count/len(y_all))
    return 


def SCtrain(model,dataset,optimizer,config):


    f1_record=[]
    loss_record=[]

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    for epoch in range(config.epochs):

        model.train()
        model.zero_grad()
        total_loss=0
        print("-------train epoch:{}-------".format(epoch))
        for x, y, batch_len in tqdm(dataloader):
            x= x.to(config.device)
            y =y.to(config.device)
            x=x.T[:max(batch_len)].T
            batch_len,indices=batch_len.sort(descending=True)
            y=y[indices]
            x=x[indices]
            loss = model(x,y)
            total_loss+=float(loss)
            loss.backward()
            optimizer.step()
            model.zero_grad()
        torch.save(model.state_dict(), config.model_path)
        print("evaluating...")
        SCevaluate(model,dataset,dataset.tag2num,config)
        print("loss: %.4f" % (total_loss))
        loss_record.append(total_loss)

    return 



class SCConfig():
    """
    参数
    """
    def __init__(
            self,
            batch_size=32,
            epochs=20,
            device="cuda:1",
            print_tfn=False, #是否输出TP,FP,FN
            PRF_mode="micro", #确定采用微平均还是宏平均计算PRF
            model_path="model/save/a.model"

    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.device=device
        self.print_tfn=print_tfn
        self.PRF_mode=PRF_mode
        self.model_path=model_path

