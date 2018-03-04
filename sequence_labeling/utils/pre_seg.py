# -*- coding: utf-8 -*-

import re
from . import full_half_converter


# 初分词，按标点切分
def process(sentence):
    sentence = full_half_converter.convert(sentence,
                                           full_half_converter.FH_ALPHA,
                                           full_half_converter.FH_NUM,
                                           full_half_converter.FH_SPACE)
    return list(filter(lambda sentence: sentence != "",
                       map(lambda sentence: sentence.strip(), re.split("[,，.。!！?？;；]", sentence))))
