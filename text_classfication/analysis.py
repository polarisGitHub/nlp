import codecs
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from collections import Counter

# 测试文件，预测结果
version = "base"
test_file = "data/train_data_20180115_15_29_25_test.csv"
prediction_file = "runs/" + version + "/prediction.csv"
label_file = "2018_01_02_label.csv"

test_map, prediction_map = {}, {}
label_set = set()


def read_file(file, data, splitor):
    with codecs.open(file, "r", encoding="utf-8") as f:
        for line in f:
            sentence, label = line.split(splitor)
            data[sentence.strip()] = int(float(label.strip()))
            label_set.add(int(float(label.strip())))


read_file(test_file, test_map, "__label__")
read_file(prediction_file, prediction_map, ",")

label_dict = {}
read_file(label_file, label_dict, ",")

with codecs.open("data/1", "w", encoding="utf-8") as f:
    for s, l in test_map.items():
        if s in prediction_map and prediction_map[s] != l:
            f.write(str(l) + "," + str(prediction_map[s]) + "\n")

actual, predict = [], []
for sentence, label in test_map.items():
    actual.append(label)
    predict.append(prediction_map[sentence])

cm = confusion_matrix(actual, predict)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

report = classification_report(actual, predict, target_names=["__label__" + str(i) for i in range(len(label_set))])
with codecs.open("runs/" + version + "/report.txt", "w", encoding="utf-8") as f:
    f.write(report)
