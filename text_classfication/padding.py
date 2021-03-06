# -*- coding: utf-8 -*-
import re
import codecs

max_len = 32
files = ["textcnn/data/test"]


def padding(words, length):
    if len(words) > max_len:
        return words[0:max_len]
    else:
        words.extend(["<pad>"] * (length - len(words)))
        return words


for file in files:
    with codecs.open(file + ".csv", "r", encoding="utf-8") as r:
        with codecs.open(file + "_padding.csv", "w", encoding="utf-8") as w:
            for line in r:
                sentence, label = line.split("\t")
                w.write(" ".join(padding(sentence.split(" "), max_len)) + "\t" + label)
