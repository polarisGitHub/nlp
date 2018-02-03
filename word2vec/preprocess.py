# -*- coding: utf-8 -*-

import os
import codecs
import collections
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
parser.add_option("-c", "--min_count",
                  dest="min_count",
                  default=5,
                  help="word will replace to __UNK__, if word_count less than min_count,default is 5")
parser.add_option("-u", "--unk",
                  dest="unk",
                  default="__UNK__",
                  help="unk symbol,default is __UNK__")
(options, args) = parser.parse_args()

if options.input == "":
    parser.error("input is empty")
if options.output == "":
    parser.error("output is empty")

options.min_count = int(options.min_count)
# 统计词频
lines = 0
counter = collections.Counter()
with codecs.open(options.input, "r", encoding="utf-8") as input_file:
    for line in input_file:
        for word in line.strip(os.linesep).split(" "):
            counter.update({word: 1})
        lines += 1
        if lines % 5000 == 0:
            print("read line %s" % lines)

# 重写unk
processed = 0
with codecs.open(options.input, "r", encoding="utf-8") as input_file:
    with codecs.open(options.output, "w", encoding="utf-8") as output_file:
        for line in input_file:
            replaced_words = []
            for word in line.strip(os.linesep).split(" "):
                if word.strip(" ") == "":
                    continue
                if counter[word] < options.min_count:
                    replaced_words.append(options.unk)
                else:
                    replaced_words.append(word)
            output_file.write("%s%s" % (" ".join(replaced_words), os.linesep))
            processed += 1
            if processed % 5000 == 0:
                print("processed %f" % (processed / lines))
