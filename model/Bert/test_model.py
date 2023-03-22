from transformers import BertConfig


config=BertConfig.from_pretrained("bert-base-chinese",num_labels=3,label2id="hehe",lll=1)

print(config.num_labels)
print(config.label2id)
print(config.lll)