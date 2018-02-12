# -*- coding: utf-8 -*-

import codecs
import pandas as pd
from optparse import OptionParser

parser = OptionParser("usage: %prog [options] arg1 arg2")
parser.add_option("-i", "--input",
                  dest="input",
                  default="",
                  help="input file")
parser.add_option("-o", "--optput",
                  dest="optput",
                  default="",
                  help="input file")
(options, args) = parser.parse_args()
if options.input == "":
    parser.error("input is empty")
if options.optput == "":
    parser.error("optput is empty")

data = pd.read_csv(options.input, sep="\t", header=None)

sentence_cache = dict()

for index, row in data.iterrows():
    sentence, label = row[0].strip(), row[1].strip()
    if sentence not in sentence_cache:
        sentence_cache[sentence] = []
    sentence_cache[sentence].append(label)

with codecs.open(options.optput, "w", encoding="utf-8") as w:
    for sentence, labels in sentence_cache.items():
        w.write(sentence + "\t" + ",".join(labels) + "\n")
