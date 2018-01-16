import codecs
import pandas as pd
import matplotlib.pyplot as plt

data_frame = {"sentence": [], "label": []}
examples = list(codecs.open("data/2018_01_02_all_train_data.csv", "r", encoding="utf-8").readlines())
for item in examples:
    data, label = item.split("__label__")
    label = int(label.rstrip("\n"))
    data_frame["sentence"].append(data.rstrip("\t"))
    data_frame["label"].append(label)

df = pd.DataFrame(data=data_frame)
label_group = df.groupby("label")

# 统计
# x, y = [], []
# for i in range(label_group.ngroups):
#     cnt = label_group.get_group(i)["sentence"].count()
#     x.append(i)
#     y.append(cnt)
# plt.bar(x=x, height=y)
# plt.show()

test_ratio = 0.2
train, test = {}, {}
