import torch
from tqdm import tqdm

class StandardEvaluator:
    """
    标准测试框架
    需要重写3个函数
    """

    def __init__(self):
        """对结果进行初始化"""
        self.y_all=[]
        self.y_hat_all=[]

    def add(self,y,y_hat):
        """添加记录"""
        pass

    def calculate_PRF(self):
        return 0,0,0,0


    def labelize(self,dataloader,model):
        """
        没有评估的操作，只有标注
        
        """
        model.eval()
        print("labeling...")
        with torch.no_grad():
            for x, y in tqdm(dataloader):  # 这里不一定是三个参数
                y_hat = model(x, y)
                # 记录
                self.add(y,y_hat)

    def evaluate(self,dataloader,model):
        """
        程序入口
        要求：
            参数已经配置好，包括：dataloader,model
        流程：
            初始化变量，y和y_hat，用来
            模型预测
            处理并保存中间变量
            评估

        """
        model.eval()
        print("evaluating...")
        with torch.no_grad():
            for x, y in tqdm(dataloader):  # 这里不一定是三个参数
                y_hat = model(x, y)
                # 记录
                self.add(y,y_hat)
        # 评估
        acc, p, r, f1 = self.calculate_PRF()

        # 返回
        return acc, p, r, f1
