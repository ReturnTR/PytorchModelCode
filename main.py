from torch.utils.data import DataLoader
from .CommonTools.JsonTool import save_json
import torch
import os

def train_and_evaluate(
        dataset,
        model,
        save_model_path,
        trainer,
        evaluator,
        optimizer,
        collate_fn,
        epochs=20,
        batch_size=32,
        shuffle=False,
        mode="train",
        dev_save_path=""
):

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )


    if dev_save_path!="":
        """保存记录文件"""
        print("labeling dev file...")
        dataset.set_mode("test")
        model.load_state_dict(torch.load(save_model_path))
        evaluator.labelize(dataloader, model)
        y_hat_all=evaluator.y_hat_all
        x=dataset.dev_data
        res=[["".join(dataset.dev_data_without_tokenize[i]),dataset.dev_labels[i],evaluator.y_hat_all[i]] for i in range(len(y_hat_all))]
        save_json(res,dev_save_path)
        return 


    # 提供测试功能
    if mode=="test":
        
        dataset.set_mode("test")
        if os.path.exists(save_model_path):
            print("load model from path")
            model.load_state_dict(torch.load(save_model_path))
        else:
            print("training test_data...")
            loss=trainer.train(model,dataloader,optimizer)
            torch.save(model.state_dict(), save_model_path)
            print("loss: %.4f" % (loss))
        print("-----test evaluate-----")
        evaluator.evaluate(dataloader, model)
        print("-----dev evaluate-----")
        dataset.set_mode("dev")
        evaluator.evaluate(dataloader, model)

    elif mode=="train":

        f1_record=[]
        loss_record=[]
        for epoch in range(epochs):
            dataset.set_mode("train")

            print("-------train epoch:{}-------".format(epoch))
            loss=trainer.train(model,dataloader,optimizer)
            print("loss: %.4f" % (loss))
            loss_record.append(loss)
            torch.save(model.state_dict(), save_model_path)
            print("-----train-----")
            evaluator.evaluate(dataloader,model)
            print("-----test-----")
            dataset.set_mode("test")
            evaluator.evaluate(dataloader, model)
            print("-----dev-----")
            dataset.set_mode("dev")
            evaluator.evaluate(dataloader, model)
            
            if len(f1_record) > 10:
                if f1_record[-1] <= f1_record[-6]:
                    print("dev的f1不再下降，训练结束")
                    break