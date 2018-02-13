# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("textcnn/runs/1518353084/softmax.csv", sep=",", header=None)

sentence, label = data[0], data[1]
softmax = list(map(lambda l: np.fromstring(l, dtype=np.float16, sep=","), data[2]))

fuzzy_sentence, top1, top2, prob1, prob2 = [], [], [], [], []
for i in range(len(sentence)):
    prob = softmax[i]
    top = sorted(range(len(prob)), key=lambda index: prob[index], reverse=True)[:2]
    top_prob = prob[top]
    if top[1] == label[i]:
        fuzzy_sentence.append(sentence[i].replace("<pad>", "").strip())
        top1.append(top[0])
        top2.append(top[1])
        prob1.append(top_prob[0])
        prob2.append(top_prob[1])

fuzzy_sentence_column, top1_column, top2_column = pd.Series(fuzzy_sentence), pd.Series(top1), pd.Series(top2)
prob1_column, prob2_column = pd.Series(prob1), pd.Series(prob2)
save = pd.DataFrame({"fuzzy": fuzzy_sentence_column, "top1": top1_column, "top2": top2_column, "prob1": prob1_column,
                     "prob2": prob2_column})
save.to_csv('b.txt', header=False, index=False, sep=',', encoding="utf-8")

sns.set()
sns.jointplot("x", "y", pd.DataFrame({"x": prob1, "y": prob2}), kind='hex')
plt.show()
