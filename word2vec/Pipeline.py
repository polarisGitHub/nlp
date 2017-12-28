# -*- coding: utf-8 -*-

import codecs


class Char:
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for file in self.files:
            for line in codecs.open(file, "utf-8"):
                yield line.split()
