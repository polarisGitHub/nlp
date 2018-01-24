# -*- coding: utf-8 -*-

import codecs


class SplitWord:
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for file in self.files:
            for line in codecs.open(file, "r", encoding="utf-8"):
                words = [item for item in line.strip().split(" ") if item.strip() != ""]
                yield words
