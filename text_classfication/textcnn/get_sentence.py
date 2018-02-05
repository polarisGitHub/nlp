# -*- coding: utf-8 -*-

import codecs

with codecs.open("data/train_padding.csv", "r", encoding="utf-8") as r:
    with codecs.open("data/train_padding_sentence.csv", "w", encoding="utf-8") as w:
        for line in r:
            sentence, _ = line.split("\t")
            w.write(sentence + "\n")
