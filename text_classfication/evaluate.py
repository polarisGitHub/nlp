# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from optparse import OptionParser

parser = OptionParser("usage: %prog [options] arg1 arg2")
parser.add_option("-s", "--softmax",
                  dest="softmax",
                  default="",
                  help="input file")
parser.add_option("-t", "--test",
                  dest="test",
                  default="",
                  help="test file")

(options, args) = parser.parse_args()
if options.softmax == "":
    parser.error("softmax is empty")
if options.test == "":
    parser.error("test is empty")

predict_top_k = 3


def convert_label_to_int(label):
    return int(label.strip().replace("__label__", ""))


def actual_predict(actual, softmax, top_k=3):
    top = sorted(range(len(softmax)), key=lambda i: softmax[i], reverse=True)[:top_k]
    return actual in top


# 不是每个句子的分类都是清晰的，定义模糊分类
def vague_predict(actual, softmax):
    top = sorted(range(len(softmax)), key=lambda i: softmax[i], reverse=True)[:3]
    p = softmax[top]
    if p[0] - p[1] < 0.2:
        return False, actual in top[0:2]
    elif p[0] < 0.8 or p[1] > 0.15:
        return True, actual in top
    else:
        return False, actual == top[0]


# read test data
test_csv = pd.read_csv(options.test, sep="\t", header=None)
test_sentence = test_csv[0]
test_label = test_csv[1]

# read softmax
softmax_csv = pd.read_csv(options.softmax, sep=",", header=None)
test_softmax = list(map(lambda l: np.fromstring(l, dtype=np.float16, sep=","), softmax_csv[1]))

total = len(test_label)
actual_precise_cnt = [0] * predict_top_k  # 准确意义下，正确率
vague_cnt = 0  # 模糊率
vague_precise_cnt = 0  # 模糊意义下，正确率
for i in range(total):
    actual_label = convert_label_to_int(test_label[i])
    for j in range(1, predict_top_k + 1):
        if actual_predict(actual_label, test_softmax[i], top_k=j):
            actual_precise_cnt[j - 1] += 1
    is_vague, is_vague_right = vague_predict(actual_label, test_softmax[i])
    if is_vague:
        vague_cnt += 1
    if is_vague_right:
        vague_precise_cnt += 1

print("actual", list(map(lambda l: l / total, actual_precise_cnt)))
print("vague", vague_cnt / total, vague_precise_cnt / total)
