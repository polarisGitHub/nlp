# -*- coding: utf-8 -*-

import re


# 初分词，按标点切分
def process(sentence):
    return list(filter(lambda sentence: sentence != "",
                       map(lambda sentence: sentence.strip(), re.split("[,，.。!！?？;；]", sentence))))
