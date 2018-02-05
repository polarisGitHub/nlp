# -*- coding: utf-8 -*-
import re
import codecs

max_len = 64
files = ["train", "test"]

regexes = {
    re.compile("(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]"): "URLTOKEN",
    re.compile("\d{17}[\d|x]|\d{15}"): "IDCARDTOKEN",
    re.compile("jd_[\d\w/*]*"): "PINTOKEN",
    re.compile("#E-s(\d)+"): "STICKERTOKEN",
    re.compile("(13|14|15|17|18|19)([0-9|*]){9}"): "MOBILETOKEN",
    re.compile("^(0[0-9]{2,3}/-)?([2-9][0-9]{6,7})+(/-[0-9]{1,4})}"): "PHONETOKEN",
    re.compile("[^_][a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)+"): "EMAILTOKEN",
    re.compile("[a-zA-Z0-9_-|*]+@[a-zA-Z0-9_-|*]+"): "EMAILTOKEN",
    re.compile("[-]?[0-9]*\.?[0-9]+"): "NUMTOKEN"
}


def replace(text):
    for patten, token in regexes.items():
        text = patten.sub(token, text)
    return text


def padding(words, length):
    if len(words) > max_len:
        return words[0:max_len]
    else:
        words.extend(["<pad>"] * (length - len(words)))
        return words


for file in files:
    with codecs.open("data/" + file + ".csv", "r", encoding="utf-8") as r:
        with codecs.open("data/" + file + "_padding.csv", "w", encoding="utf-8") as w:
            for line in r:
                sentence, label = line.split("\t")
                w.write(" ".join(padding(replace(sentence).split(" "), max_len)) + "\t" + label)
