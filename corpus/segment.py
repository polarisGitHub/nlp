# -*- coding: utf-8 -*-

import codecs
from word_cut import WordCut
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
parser.add_option("-u", "--user_dict",
                  dest="user_dict",
                  default=None,
                  help="user dict")
parser.add_option("-s", "--stop_dict",
                  dest="stop_dict",
                  default=None,
                  help="stop dict")

(options, args) = parser.parse_args()
if options.input == "":
    parser.error("input is empty")
if options.output == "":
    parser.error("output is empty")

index = 0
word_cut = WordCut(user_dict=options.user_dict, stop_dict=options.stop_dict)
with codecs.open(options.input, "r", encoding="utf-8") as r:
    with codecs.open(options.output, "w", encoding="utf-8") as w:
        for line in r:
            w.write(" ".join(word_cut.seg(line)))
            index += 1
            if index % 500 == 0:
                print(index)
