from .Standard import StandardEvaluator
from ..CommonTools.BasicTool import DictCount
from ..CommonTools.MathTool import PGR2PRF
class SCEvaluator(StandardEvaluator):

    """
    句子分类的评估方法
    """

    def __init__(self,output_value=False):
        super().__init__()
        self.y_all=[]
        self.y_hat_all=[]
        self.output_value=output_value

    def add(self, y, y_hat):
        self.y_all+=[int(i) for i in y]
        if self.output_value:
            # 输出某个类别的值，不输出结果
            self.y_hat_all+=[float(i) for i in y_hat]
        else:
            self.y_hat_all+=[int(i) for i in y_hat.indices]


    def calculate_PRF(self):
        if len(self.y_all)!=len(self.y_hat_all):
            print("长度不匹配，无法计算！")
            return

        right=DictCount()
        predict=DictCount()
        gold=DictCount()
        for i in range(len(self.y_all)):
            gold.add(self.y_all[i])
            predict.add(self.y_hat_all[i])
            if self.y_all[i]==self.y_hat_all[i]:
                right.add(self.y_all[i])
        right=right.get()
        predict=predict.get()
        gold=gold.get()

        PRF=dict()

        # 记录每一个类别的PRF
        for label in gold.keys():
            if label not in predict:
                predict[label]=0
                print(label, "不在预测范围内")
            if label not in right:
                right[label]=0
                print(label,"不在正确预测范围内")
                PRF[label]={"p":-1,"r":-1,"f":-1}
            else:
                p,r,f=PGR2PRF(gold[label],predict[label],right[label])
                PRF[label]={"p":p,"r":r,"f":f}
        print(PRF)
        return 0, 0, 0, 0