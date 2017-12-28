# -*- coding: utf-8 -*-

import codecs


class Character:
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for file in self.files:
            for line in codecs.open(file, "r", encoding="utf-8"):
                yield [_ for _ in line]
