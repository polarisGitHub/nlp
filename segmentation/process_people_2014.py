# -*- coding: utf-8 -*-

import os
import re
import codecs

from utils import file_walker
from utils import pre_seg

sentences = []
print("开始处理")
for file in file_walker.find_files(os.getcwd() + "/data/2014"):
    with codecs.open(file, "r", encoding="utf-8") as f:
        words = re.findall(
            "([\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\u4e00-\u9fa5\w]+)/[a-z]+",
            f.readline().strip())
        pre_process_sentence = pre_seg.process(" ".join(words))  # 按标点预分割
        for sentence in pre_process_sentence:
            sentences.append(sentence)
print("处理完毕")

print("开始写文件")
with codecs.open("data/2014_process/word_cut.txt", "w", encoding="utf-8") as f:
    f.write("\r".join(sentences))
print("写文件完毕")
