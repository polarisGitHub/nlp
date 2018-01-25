# -*- coding: utf-8 -*-

import re

# name
# datetime
# address

regexes = {
    re.compile("(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]"): "__URL__",
    re.compile("\d{17}[\d|x]|\d{15}"): "__IDCARD__",
    re.compile("jd_[\d\w/*]*"): "__PIN__",
    re.compile("#E-s(\d)+"): "__STICKER__",
    re.compile("(13|14|15|17|18|19)([0-9|*]){9}"): "__MOBILE__",
    re.compile("^(0[0-9]{2,3}/-)?([2-9][0-9]{6,7})+(/-[0-9]{1,4})}"): "__PHONE__",
    re.compile("[^_][a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)+"): "__MAIL__",
    re.compile("[a-zA-Z0-9_-|*]+@[a-zA-Z0-9_-|*]+"): "__MAIL__",
    re.compile("[-]?[0-9]*\.?[0-9]+"): "__NUM__"
}


def replace(text):
    for patten, token in regexes.items():
        text = patten.sub(token, text)
    return text
