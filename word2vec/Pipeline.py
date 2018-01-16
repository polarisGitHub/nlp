# -*- coding: utf-8 -*-

import jieba
import codecs


class Character:
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for file in self.files:
            for line in codecs.open(file, "r", encoding="utf-8"):
                yield [_ for _ in line]


class Word:
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for file in self.files:
            for line in codecs.open(file, "r", encoding="utf-8"):
                yield [item for item in line if item.strip() != ""]
