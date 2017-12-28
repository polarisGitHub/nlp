# -*- coding: utf-8 -*-

# 解压的wiki有多个顶层元素，xml解析错误
# 该程序给每个wiki添加一个顶层元素
import os
import re
import codecs
import collections
import numpy as np

# wiki中有(),（。）,把这些类型都去除
remove_token = {"\{[^\}]?\}",
                "（[^）]?）", "\([^\)]?\)", "（[^\)]?\)", "（[^\)]?\)",
                "【[^】]?】", "\[[^\]]?\]", "【[^\]]?\]", "【[^\]]?\]",
                "「[^」]?」", "『[^』]?』", "『[^」]?」", "『[^」]?」",
                "《[^》]?》", "<[^>]?>", "《[^>]?>", "《[^>]?>"}


def process_data(data):
    for token in remove_token:
        data = re.sub(token, "", data)
    return data


files = []
for file in os.listdir("./data"):
    if file.startswith("wiki"):
        files.append(file)

for f in files:
    with codecs.open("./data/extract_wiki", "w", encoding="utf-8") as extract:
        with codecs.open("./data/" + f, "r", encoding='utf-8') as file:
            for line in file:
                # 去除xml标记和空行
                if line.startswith("<doc") or line.startswith("</doc>") or re.match("^\s*$", line):
                    continue
                extract.write(line)
