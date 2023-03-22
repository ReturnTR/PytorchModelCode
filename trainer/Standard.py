
from tqdm import tqdm
class StandardTrainer:

    @staticmethod
    def train(model,dataloader,optimizer):
        model.train()
        model.zero_grad()
        total_loss = 0
        for x, y in tqdm(dataloader):
            loss = model(x, y)
            total_loss += float(loss)
            loss.backward()
            optimizer.step()
            model.zero_grad()
        print("loss: %.4f" % (total_loss))
        return total_loss


