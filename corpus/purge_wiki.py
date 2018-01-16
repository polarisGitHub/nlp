# -*- coding: utf-8 -*-

import os
import re
import codecs

# wiki中有(),（。）,把这些类型都去除
remove_token = ["\{[^\}]?\}",
                "（[^）]?）", "\([^\)]?\)", "（[^\)]?\)", "（[^\)]?\)",
                "【[^】]?】", "\[[^\]]?\]", "【[^\]]?\]", "【[^\]]?\]",
                "「[^」]?」", "『[^』]?』", "『[^」]?」", "『[^」]?」",
                "《[^》]?》", "<[^>]?>", "《[^>]?>", "《[^>]?>"]


def process_data(data):
    for token in remove_token:
        data = re.sub(token, "", data)
    return data


files = []
for file in os.listdir("./data/wiki"):
    if file.startswith("wiki"):
        files.append(file)

with codecs.open("./data/corpus_wiki", "w", encoding="utf-8") as extract:
    for f in files:
        with codecs.open("./data/wiki/" + f, "r", encoding='utf-8') as file:
            for line in file:
                # 去除xml标记和空行
                if line.startswith("<doc") or line.startswith("</doc>") or re.match("^\s*$", line):
                    continue
                extract.write(process_data(line))
