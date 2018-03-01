# -*- coding: utf-8 -*-

import re
import codecs
from optparse import OptionParser

parser = OptionParser("usage: %prog [options] arg1 arg2")
parser.add_option("-i", "--input",
                  dest="input",
                  default="",
                  help="input file")
parser.add_option("-o", "--output",
                  dest="output",
                  default="",
                  help="output file")
(options, args) = parser.parse_args()
if options.input == "":
    parser.error("input is empty")
if options.output == "":
    parser.error("output is empty")

# name
# datetime
# address
regexes = {
    re.compile("(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]"): "URLTOKEN",
    re.compile("\d{4}([-/年])\d{1,2}([-/月])\d{1,2}([-/日])?"): "DATETOKEN",
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


index = 0
with codecs.open(options.input, "r", encoding="utf-8") as r:
    with codecs.open(options.output, "w", encoding="utf-8") as w:
        for line in r:
            w.write(replace(line))
            index += 1
            print(index)
