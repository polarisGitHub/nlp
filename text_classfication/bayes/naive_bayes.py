# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# 读取数据
df = pd.read_csv('data/train.csv', header=None)
train_data, train_label = df[0].tolist(), df[1].tolist()
df = pd.read_csv('data/test.csv', header=None)
test_data, test_label = df[0].tolist(), df[1].tolist()


# 统计词语出现次数
count_vect = CountVectorizer()
train_counts = count_vect.fit_transform(train_data)
# 使用tf-idf方法提取文本特征
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_counts)


# 训练
clf = MultinomialNB().fit(train_tfidf, train_label)
print("分类器的相关信息：", clf)

# 测试
test_counts = count_vect.transform(test_data)
test_tfidf = tfidf_transformer.transform(test_counts)
predicted = clf.predict(test_tfidf)
print(np.mean(predicted == test_label))
