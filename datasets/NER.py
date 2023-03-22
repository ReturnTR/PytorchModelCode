
from regex import E
from sklearn import model_selection
from torch.utils.data import Dataset,DataLoader
import torch
from tqdm import tqdm
import time
from 



def file2list(path):
    """
    将序列标注文件转化为二维列表，文件格式为每行一个token
    :param path: 文件路径
    :return:  数据和标签的二维列表
    例：train_data,train_labels=file2list(train_path)
    """

    file=open(path,'r',encoding='utf-8')
    data=file.read()
    file.close()
    data=data.strip("\n")
    data=data.split("\n\n")
    data=[item.split("\n") for item in data]
    train_data=[]
    train_labels=[]
    for i in data:
        train_data.append([])
        train_labels.append([])
        for j in i:
            line=j.split("\t")
            train_data[-1].append(line[0])
            train_labels[-1].append(line[1])
    return train_data,train_labels

def list2file(data,labels,filename):
    """
    file2list函数的功能倒过来
    """
    res=[]
    for i in range(len(data)):
        line=[]
        for j in range(len(data[i])):
            line.append(data[i][j]+"\t"+labels[i][j])
        line="\n".join(line)
        res.append(line)
    res="\n\n".join(res)
    with open(filename,'w',encoding='utf-8')as file:
        file.write(res)

def cut_BIO_file(filename,obj_path):
    """
    将数据按8：1：1切分到指定目录下
    以trian,test,dev命名
    """
    data,labels=file2list(filename)
    data=[[data[i],labels[i]]for i in range(len(data))]
    train,test,dev=cut_data(data)
    train_data=[i[0] for i in train]
    train_labels=[i[1] for i in train]
    test_data=[i[0] for i in test]
    test_labels=[i[1] for i in test]
    dev_data=[i[0] for i in dev]
    dev_labels=[i[1] for i in dev]
    list2file(train_data,train_labels,obj_path+"/train")
    list2file(test_data,test_labels,obj_path+"/test")
    list2file(dev_data,dev_labels,obj_path+"/dev")




def calculate_BIO_PRF(y_hat, y, tag2num, print_tfn, mode="micro"):
    # 记录起始标签
    B_tags = []
    I_tags = []
    tags_name=[]
    y = y.tolist()
    y_hat = y_hat.tolist()
    o = set()
    o.add(tag2num['O'])
    o.add(tag2num['<PAD>'])

    items = list(tag2num.keys())
    del items[items.index("O")]
    del items[items.index("<PAD>")]
    for b_item in items:
        b_item_split = b_item.split("-")
        if b_item_split[0] == "B":
            B_tags.append(tag2num[b_item])
            I_tags.append(tag2num["I-"+b_item_split[1]])
            tags_name.append(b_item_split[1])

    FP = [0] * len(B_tags)
    TP = [0] * len(B_tags)
    FN = [0] * len(B_tags)
    i = 0
    count = 0

    for j in y:
        if j in B_tags: count += 1
    while i < len(y):
        if y[i] in B_tags:
            style_index = B_tags.index(y[i])
            flag = 0
            if y[i] != y_hat[i]: flag = 1
            i += 1
            while i < len(y) and y[i] == I_tags[style_index]:
                if y[i] != y_hat[i]:
                    flag = 1
                i += 1
            if flag == 1:
                FN[style_index] += 1
            else:
                TP[style_index] += 1
        else:
            i += 1
    i = 0
    while i < len(y_hat):
        if y_hat[i] in B_tags:
            style_index = B_tags.index(y_hat[i])
            flag = 0
            if y[i] != y_hat[i]: flag = 1
            i += 1
            while i < len(y) and y_hat[i] == I_tags[style_index]:
                if y[i] != y_hat[i]:
                    flag = 1
                i += 1
            if flag == 1:
                FP[style_index] += 1
        else:
            i += 1

    # 计算每一个实体类型的PRF
    tag_PRF=dict()
    macro=[]
    for i in range(len(tags_name)):

        if TP[i]+FP[i]==0:
            precision=-1
            recall=-1
            F=-1
        else:
            precision = TP[i] / (TP[i] + FP[i])
            recall = TP[i] / (TP[i] + FN[i])
            F = 2 * precision * recall / (precision + recall)
        tag_PRF[tags_name[i]]={"TP":TP[i],"FP":FP[i],"FN":FN[i],"P":precision,"R":recall,"F":F}
        macro.append([precision,recall,F])
    
    print("各类实体的PRF")
    for k,v in tag_PRF.items():
        print(k,":",v)


    # 计算macro PRF
    precision=[i[0] for i in macro]
    recall=[i[1] for i in macro]
    F=[i[2] for i in macro]
    macro={"P":sum(precision)/len(precision),"R":sum(recall)/len(recall),"F":sum(F)/len(F)}
    print("macro:",macro)

    # 计算micro PRF
    FP_sum = sum(FP)
    TP_sum = sum(TP)
    FN_sum = sum(FN)
    if TP_sum+FP_sum==0:
        precision=-1
        recall=-1
        F=-1
    else:
        precision = TP_sum / (TP_sum + FP_sum)
        recall = TP_sum / (TP_sum + FN_sum)
        F = 2 * precision * recall / (precision + recall)

    micro={"TP":TP_sum,"FP":FP_sum,"FN":FN_sum,"P":precision,"R":recall,"F":F}

    print("micro:",micro)
    FP = torch.as_tensor(FP, dtype=torch.float32)
    TP = torch.as_tensor(TP, dtype=torch.float32)
    FN = torch.as_tensor(FN, dtype=torch.float32)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F = 2 * precision * recall / (precision + recall)

    return (precision, recall, F) if mode == "micro" else (float(torch.mean(precision)), float(
        torch.mean(recall)), float(torch.mean(F)))



def calculate_BIOES_PRF(y_hat, y, tag2num, print_tfn, mode="micro"):
    # 记录起始标签
    B_tags = []
    E_tags = []
    S_tags = []
    y = y.tolist()
    y_hat = y_hat.tolist()
    o = set()
    o.add(tag2num['O'])
    o.add(tag2num['<PAD>'])

    items = list(tag2num.keys())
    del items[items.index("O")]
    if "<PAD>" in items: del items[items.index("<PAD>")]
    for b_item in items:
        b_item_split = b_item.split("-")
        if b_item_split[0] == "B":
            B_tags.append(tag2num[b_item])
            for e_item in items:
                e_item_split = e_item.split("-")
                if e_item_split[1] == b_item_split[1] and e_item_split[0] == "E":
                    E_tags.append(tag2num[e_item])
                    break
            for s_item in items:
                s_item_split = s_item.split("-")
                if s_item_split[1] == b_item_split[1] and s_item_split[0] == "S":
                    S_tags.append(tag2num[s_item])
                    break
    FP = [0] * len(B_tags)
    TP = [0] * len(B_tags)
    FN = [0] * len(B_tags)
    i = 0
    count = 0

    for j in y:
        if j in B_tags: count += 1
    while i < len(y):
        if y[i] in B_tags:
            style_index = B_tags.index(y[i])
            flag = 0
            if y[i] != y_hat[i]: flag = 1
            i += 1
            while i < len(y) and y[i] != E_tags[style_index]:
                if y[i] != y_hat[i]:
                    flag = 1
                i += 1
            if i < len(y) and y[i] != y_hat[i]:
                flag = 1
                i += 1
            if flag == 1:
                FN[style_index] += 1
            else:
                TP[style_index] += 1

        elif y[i] in S_tags:
            style_index = S_tags.index(y[i])
            if y[i] != y_hat[i]:
                FN[style_index] += 1
            else:
                TP[style_index] += 1
            i += 1
        else:
            i += 1
    i = 0
    while i < len(y_hat):
        if y_hat[i] in B_tags:
            style_index = B_tags.index(y_hat[i])
            flag = 0
            if y[i] != y_hat[i]: flag = 1
            i += 1
            while i < len(y) and y_hat[i] != E_tags[style_index]:
                if y[i] != y_hat[i]:
                    flag = 1
                i += 1
            if i < len(y) and y[i] != y_hat[i]:
                flag = 1
                i += 1
            if flag == 1:
                FP[style_index] += 1
        elif y_hat[i] in S_tags:
            style_index = S_tags.index(y_hat[i])
            if y[i] != y_hat[i]:
                FP[style_index] += 1
            i += 1
        else:
            i += 1
    if mode == "micro":
        FP = sum(FP)
        TP = sum(TP)
        FN = sum(FN)
    else:
        FP = torch.as_tensor(FP, dtype=torch.float32)
        TP = torch.as_tensor(TP, dtype=torch.float32)
        FN = torch.as_tensor(FN, dtype=torch.float32)

    if print_tfn or TP == 0:
        print("FP:{} , TP:{} , FN:{}".format(FP, TP, FN))
        if TP == 0:
            return -1, -1, -1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F = 2 * precision * recall / (precision + recall)

    return (precision, recall, F) if mode == "micro" else (float(torch.mean(precision)), float(
        torch.mean(recall)), float(torch.mean(F)))

def calculate_PRF(y_hat, y, tag2num,print_tfn=False, mode="micro"):
    right = 0
    for i in range(len(y)):
        if y[i] == y_hat[i]: right += 1
    acc = right / len(y)
    use_BIOES = False
    for tag in tag2num:
        if tag[0] == "E" or tag[0] == "S" or tag[0] == "M":
            use_BIOES = True
            break
    p, r, f1 = calculate_BIOES_PRF(y_hat, y, tag2num,print_tfn=print_tfn, mode=mode) if use_BIOES \
    else calculate_BIO_PRF(y_hat, y, tag2num, mode=mode,print_tfn=print_tfn)
    return acc,p,r,f1

def NER_show_result(model,dataset,config):
    res=[]
    model.eval()
    num2tag=dataset.num2tag
    word_tokenizer=dataset.word_tokenizer
    dataset.set_mode("test")
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for x, y, batch_len in dataloader:
            x= x.to(config.device)
            y =y.to(config.device)
            # 缩减长度
            x=x.T[:max(batch_len)].T
            y=y.T[:max(batch_len)].T
            batch_len,indices=batch_len.sort(descending=True)
            y=y[indices]
            x=x[indices]
            y_hat = model(x,y) #batch_len * max_sentence_len

            for i in range(len(x)):
                res.append([])
                for j in range(len(x[i])):
                    res[-1].append(" ".join([word_tokenizer.decode(x[i][j]),num2tag[y[i][j]],num2tag[y_hat[i][j]]]))
    
    save_json(res,"data/CCKS2022_Wiki/dev_result.json")

def NER_label(model,dataset,config,filename):
    dataset.make_text(filename)
    data=get_json(filename)
    data=[{"sentence":i} for i in data]
    for item in data:item["model"]=[]
    model.eval()
    num2tag=dataset.num2tag
    word_tokenizer=dataset.word_tokenizer
    dataset.set_mode("label")
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    label_result=[]
    with torch.no_grad():
        for x ,batch_len,sentence_idx in tqdm(dataloader):
            x= x.to(config.device)
            # 缩减长度
            x=x.T[:max(batch_len)].T
            batch_len,indices=batch_len.sort(descending=True)
            x=x[indices]
            y_hat = model(x,None) #batch_len * max_sentence_len
            for i in range(len(x)):
                res=[]
                for j in range(len(x[i])):
                    if word_tokenizer.decode(x[i][j])not in ["[ S E P ]","[ P A D ]","[ C L S ]"]:
                        res.append([dataset.label_sentences_without_tokenize[sentence_idx[i]][j-1],num2tag[y_hat[i][j]]])
                data[sentence_idx[i]]["model"].append(res)

    save_json(data,"data/person_attribute_NER/label_result_para.json")




def NERevaluate(model,dataloader,tag2num,config):
    y_hat_all = torch.as_tensor([]).to(config.device)
    y_all = torch.as_tensor([]).to(config.device)
    model.eval()
    # 获取y和y_hat
    with torch.no_grad():
        for x, y, batch_len in dataloader:
            x= x.to(config.device)
            y =y.to(config.device)
            # 缩减长度
            x=x.T[:max(batch_len)].T
            y=y.T[:max(batch_len)].T
            batch_len,indices=batch_len.sort(descending=True)
            y=y[indices]
            x=x[indices]
            y_hat = model(x,y)
            y_hat=y_hat.contiguous().view(-1)
            y=y.contiguous().view(-1)
            y_all = torch.cat((y_all, y), 0)
            y_hat_all = torch.cat((y_hat_all, y_hat), 0)
    #去掉0
    not_mask=torch.nonzero(y_all).view(-1).tolist()
    y_all=y_all[not_mask]
    y_hat_all=y_hat_all[not_mask]

    # 评估
    acc,p,r,f1=calculate_PRF(y_hat_all, y_all, tag2num, print_tfn=config.print_tfn, mode=config.PRF_mode)
    
    return acc,p,r,f1

def NERtrain(model,dataset,optimizer,config):
    """
    适用于NER的训练程序
    需要创建model，dataset，optmizer，和NERConfig
    """
    f1_record=[]
    loss_record=[]

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    for epoch in range(config.epochs):
        dataset.set_mode("train")
        model.train()
        model.zero_grad()
        total_loss=0
        print("-------train epoch:{}-------".format(epoch))
        for x, y, batch_len in tqdm(dataloader):
            x= x.to(config.device)
            y =y.to(config.device)
            x=x.T[:max(batch_len)].T
            y=y.T[:max(batch_len)].T
            batch_len,indices=batch_len.sort(descending=True)
            y=y[indices]
            x=x[indices]
            loss = model(x,y)
            total_loss+=float(loss)
            loss.backward()
            optimizer.step()
            model.zero_grad()
        print("loss: %.4f" % (total_loss))
        loss_record.append(total_loss)
        model.eval()
        tag2num=dataset.get_tag2num()
        print("-----train-----")
        acc,p,r,f1=NERevaluate(model,dataloader,tag2num,config=config)
        dataset.set_mode("dev")
        model.eval()
        print("-----dev-----")
        acc,p,r,f1=NERevaluate(model,dataloader,tag2num,config=config)
        f1_record.append(f1)
        dataset.set_mode("test")
        model.eval()
        print("-----test-----")
        NERevaluate(model,dataloader,tag2num,config=config)
        torch.save(model.state_dict(), config.model_path)
        if len(f1_record) > 10:
            if f1_record[-1] <= f1_record[-6]:
                print("dev的f1不再下降，训练结束")
                break
    return f1_record,loss_record

def calculate_attribute2entities2num(tokens,labels):
    """
    返回二维字典：属性：实体：实体数量
    """
    attribute2entities2num=dict()
    for sentence_idx in range(len(tokens)):
        sentence_token=tokens[sentence_idx]
        sentence_label=labels[sentence_idx]
        sentence_len=len(sentence_token) # 当前这句话的长度
        index=0
        while(index<sentence_len):
            if sentence_label[index][0] in "BE":
                attribute=sentence_label[index].split('-')[1] #获取属性
                entity=sentence_token[index] #获取实体ide第一个字
                index+=1
                while(index<sentence_len and sentence_label[index][0] in "IME"): # 这里为了适应BIOES和BIOMS
                    entity+=sentence_token[index]
                    index+=1
                # 此时index在"BES"上，当前的attribute和entity为抽出的属性和实体
                if attribute not in attribute2entities2num.keys():
                    attribute2entities2num[attribute]=dict()
                if entity not in attribute2entities2num[attribute].keys():
                    attribute2entities2num[attribute][entity]=0
                attribute2entities2num[attribute][entity]+=1
            else:
                index+=1
    return attribute2entities2num

class NERConfig():
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

class NERDataset(Dataset):
    """
    NER的dataset，用于训练数据
    """
    def __init__(self,
                 data_path,
                 bert_model_path="",
                 tag_vocab_path="tags_vocab.txt",
                 word_vocab_path="word_vocab.txt",
                 enabled_max_len=200
                 ):

        
        train_path = data_path + "/train"
        test_path = data_path + "/test"
        dev_path = data_path + "/dev"
        import os
        if not os.path.exists(dev_path):
            dev_path = test_path

        # 获取源数据并转化为二维列表
        train_data, train_labels = file2list(train_path)

        # 若只有train则将数据集进行切分
        if not os.path.exists(test_path):
            train_data, train_labels, test_data, test_labels, dev_data, dev_labels = cut_data(train_data,train_labels)
        else:
            test_data, test_labels = file2list(test_path)
            dev_data, dev_labels = file2list(dev_path)


        # 保存
        self.train_data_without_tokenize=train_data
        self.train_labels_without_tokenize=train_labels
        self.test_data_without_tokenize=test_data
        self.test_labels_without_tokenize=test_labels
        self.dev_data_without_tokenize=dev_data
        self.dev_labels_without_tokenize=dev_labels

        



        # 建立tag字典
        num2tag = build_vocab(train_labels, save_path=tag_vocab_path)
        num2tag = ["<PAD>"] + num2tag
        tag2num = list2dict(num2tag)
        # tokenize
        if bert_model_path!="":
            from transformers import BertTokenizer
            word_tokenizer = BertTokenizer.from_pretrained(bert_model_path)
            tokenize(train_data, tokenizer=word_tokenizer, mode="bert")
            tokenize(test_data, tokenizer=word_tokenizer, mode="bert")
            tokenize(dev_data, tokenizer=word_tokenizer, mode="bert")
        else:
            num2word=build_vocab(train_data,save_path=word_vocab_path)
            num2word = ["<PAD>"] + num2word
            self.word_vocab_size=len(num2word)
            word_tokenizer=list2dict(num2word)
            self.word2num=word_tokenizer
            del_no_word_sentences(test_data,test_labels,word_tokenizer)
            del_no_word_sentences(dev_data,dev_labels,word_tokenizer)
            tokenize(train_data, tokenizer=word_tokenizer, mode="dict")
            tokenize(test_data, tokenizer=word_tokenizer, mode="dict")
            tokenize(dev_data, tokenizer=word_tokenizer, mode="dict")

        #这里需要手动设置是否首尾两边填0
        tokenize(train_labels, tokenizer=tag2num, mode="dict",end_pad=True)
        tokenize(test_labels, tokenizer=tag2num, mode="dict",end_pad=True)
        tokenize(dev_labels, tokenizer=tag2num, mode="dict",end_pad=True)
        # 去掉过长句子
        cut_sentence(train_data, train_labels, enabled_max_len)
        cut_sentence(test_data, test_labels, enabled_max_len)
        cut_sentence(dev_data, dev_labels, enabled_max_len)

        # 获取句长
        train_len = [len(i) for i in train_data]
        test_len = [len(i) for i in test_data]
        dev_len = [len(i) for i in dev_data]

        # padding
        padding(train_data, train_labels)
        padding(test_data, test_labels)
        padding(dev_data, dev_labels)

        # 转换成tensor
        train_data = torch.as_tensor(train_data).long()
        train_labels = torch.as_tensor(train_labels)
        test_data = torch.as_tensor(test_data).long()
        test_labels = torch.as_tensor(test_labels)
        dev_data = torch.as_tensor(dev_data).long()
        dev_labels = torch.as_tensor(dev_labels)


        self.word_tokenizer=word_tokenizer
        self.train_data = train_data
        self.train_labels = train_labels
        self.train_len = train_len
        self.test_data = test_data
        self.test_labels = test_labels
        self.test_len = test_len
        self.dev_labels = dev_labels
        self.dev_data = dev_data
        self.dev_len = dev_len
        self.tag2num = tag2num
        self.num2tag = num2tag
        print(num2tag)

    def get_word_vocab_size(self):
        return self.word_vocab_size

    def get_tag2num(self):
        return self.tag2num

    def get_word2num(self):
        return self.word2num

    def set_mode(self, mode):
        self.mode = mode

    def get_no_tokenize_data(self):
        if self.mode == "train":
            return self.train_data_without_tokenize,self.train_labels_without_tokenize
        elif self.mode == "test":
            return self.test_data_without_tokenize,self.test_labels_without_tokenize
        elif self.mode == "dev":
            return self.dev_data_without_tokenize,self.dev_labels_without_tokenize
        else:
            raise ("Wrong mode!")       


    def __getitem__(self, item):
        if self.mode == "train":
            return self.train_data[item], self.train_labels[item], self.train_len[item]
        elif self.mode == "test":
            return self.test_data[item], self.test_labels[item], self.test_len[item]
        elif self.mode == "dev":
            return self.dev_data[item], self.dev_labels[item], self.dev_len[item]
        elif self.mode=="label":
            return self.label_sentences[item],self.label_sentences_len[item],self.sentence_index[item]
        else:
            raise ("Wrong mode!")

    def __len__(self):
        if self.mode == "train":
            return len(self.train_data)
        elif self.mode == "test":
            return len(self.test_data)
        elif self.mode == "dev":
            return len(self.dev_data)
        elif self.mode=="label":
            return len(self.label_sentences)
        else:
            raise ("Wrong mode!")


    def make_text(self,filename):
        """
        data:sentence列表，可能是二维
        用模型标注数据
        """
        data=get_json(filename)
        sentence2index=[]
        sentences=[]
        idx=0
        for para in data:
            para=para["summary"]
            # para=para.split("。")
            # para=[i for i in para if i]
            # para=[i.strip() for i in para]
            # para=[i for i in para if i]
            # para=[i+"。" for i in para if i]
            para="。".join(para)
            sentences.append(para)
            sentence2index.append(idx)
            idx += 1
        sentences=[list(i) for i in sentences]
        import copy

        cut_sentence(sentences, None, 500)
        self.label_sentences_without_tokenize = copy.deepcopy(sentences)
        self.label_sentences_without_tokenize=[i+[0]*500 for i in self.label_sentences_without_tokenize]
        tokenize(sentences, tokenizer=self.word_tokenizer, mode="bert")
        sentences_len = [len(i) for i in sentences]
        self.label_sentences_len=sentences_len
        padding(sentences)
        sentences = torch.as_tensor(sentences).long()
        self.label_sentences=sentences
        self.sentence_index=sentence2index

class NERDataAnalysis():
    """
    分析NER数据集
    """
    def __init__(self,file_path) -> None:
        self.tokens,self.labels=file2list(file_path)
        self.num2token = build_vocab(self.tokens, save=False)
        self.num2tag = build_vocab(self.labels, save=False)

        # NER任务类型：BIOES or BIO
        self.BIOES=False
        for tag in self.num2tag:
            if tag[0] in "SME":
                self.BIOES=True
                break
        self.attribute2entities2num=calculate_attribute2entities2num(self.tokens,self.labels)
        self.attributes=list(self.attribute2entities2num.keys())

    def get_entity_num(self):
        """
        实体：数量
        """
        entity_num=dict()
        for entities2num in self.attribute2entities2num.values():
            for entity,num in entities2num.items():
                if entity not in entity_num:
                    entity_num[entity]=0
                entity_num[entity]+=num
        return sort_dict(entity_num)

    def get_attribute_num(self):
        """
        属性：数量
        """
        attribute_num=dict()
        for attribute in self.attribute2entities2num.keys():
            attribute_num[attribute]=0
            for num in self.attribute2entities2num[attribute].values():
                attribute_num[attribute]+=num
        return sort_dict(attribute_num)
    
    def calculate_entities_num(self,entities):
        """
        :param entities 实体，可以是列表，集合，元组
        计算语料库存在这些实体的个数，没有为0
        注：该实体与与语料库中的标签无关
        注：不支持实体嵌套，以短实体优先
        返回：{实体：数量}

        """
        max_entity_len=max([len(entity) for entity in entities])

        entities=set(entities)
        entitiy_num=dict()
        for entity in entities:entitiy_num[entity]=0
        for sentence_idx in range(len(self.tokens)):
            sentence_token=self.tokens[sentence_idx]
            sentence_len=len(sentence_token)
            index=0
            while(index<sentence_len):
                temp_index=index
                word=""
                has_entity=False # 当前index下有没有实体的flag
                while(temp_index<sentence_len and temp_index-index<=max_entity_len):
                    word=word+sentence_token[temp_index]
                    if word in entities:
                        has_entity=True
                        entitiy_num[word]+=1
                        index=temp_index+1
                        break
                    else:temp_index+=1
                if not has_entity:
                    index+=1
        return sort_dict(entitiy_num)
                        
def NER_data_analysis(data_path):
    """
    分析NER数据都要看什么，来了解数据的全貌
    每个文件的数据，两个文件的交叉数据

    注：字典数据中值为数字时需要进行排序以便可视化
    每个文件的数据
        1. token种类数量（token太多不予显示）
        2. 全部的tag，这个数量少可以显示，即所有的属性，所有的属性数量，可以显示为属性：实体数量，并对其从大到小排序
        3. 文件的总实体，可以显示为：总实体数量，总实体种类，以及实体：数量，并对其从大到小排序
                此外，还有实体的出现次数分布，即1-5，6-10，10-20，...的实体占多少，这个做图更好
        4. 本身在实体词典里面，但是没有被标为实体的词语，数量
        5. 实体嵌套

    两个文件的交叉数据，用train和test来表示两个文件
        1. 实体在test中，但是没有在train中的实体（OOV）数量，总数量，种类数量，在test中的占比
        2. OOV在train中的数量
    """

    def analysis_single_file(analysis):
        print("token种类数量（词典大小）：{}".format(len(analysis.num2token)))
        print("属性数量：",end="")
        print(analysis.get_attribute_num())
        entity_num=analysis.get_entity_num()
        print("实体数量：{}".format(sum(entity_num.values())))
        print("句子数量：{}".format(len(analysis.tokens)))
        print("平均句长：{}".format(sum([len(i) for i in analysis.tokens])/len(analysis.tokens)))
        print("实体数量/句子数量：{:.2f}".format(sum(entity_num.values())/len(analysis.tokens)))
        print("实体种类数量：{}".format(len(entity_num.keys())))
        entity_num_distribution=sort_dict(num_distribution(entity_num.values()))
        print("实体出现次数分布（数量区间：实体出现次数在该区间内的实体的数量）：")
        print(entity_num_distribution)
        print("占比：") #保留三位小数
        rate={name:int(num/len(entity_num.keys())*1000)/1000 for name,num in entity_num_distribution.items()}
        print(rate)
        
    def analysis_cross_file(analysis_train,analysis_test):
        """
        注：交叉信息是不对称的
        """

        OOV=dict()

        train_entity_num=analysis_train.get_entity_num()
        test_entity_num=analysis_test.get_entity_num()
        train_entity_set=set(train_entity_num.keys())
        for entity in test_entity_num.keys():
            if entity not in train_entity_set:
                OOV[entity]=test_entity_num[entity]
        print("OOV种类：{}".format(len(OOV.keys())))
        print("OOV数量：{}".format(sum(OOV.values())))
        print("OOV实体出现次数的分布：")
        entity_num_distribution=sort_dict(num_distribution(OOV.values(),distribution_num=[0,1,2,3,10,50,100]))
        print(entity_num_distribution)
        print("占比：")
        rate={name:int(num/len(OOV.keys())*1000)/1000 for name,num in entity_num_distribution.items()}
        print(rate)


    train_data_path=data_path+"/train"
    test_data_path=data_path+"/test"
    dev_data_path=data_path+"/dev"
    train_data=NERDataAnalysis(train_data_path)
    test_data=NERDataAnalysis(test_data_path)
    dev_data=NERDataAnalysis(dev_data_path)
    print("-------train_file-------")
    analysis_single_file(train_data)
    print("-------test_file--------")
    analysis_single_file(test_data)
    print("-------dev_file--------")
    analysis_single_file(dev_data)
    print("-------train-test OOV-information------")
    analysis_cross_file(train_data,test_data)
    print("-------train-dev OOV-information------")
    analysis_cross_file(train_data,dev_data)

