# -*- coding: utf-8 -*-

import gensim
from optparse import OptionParser

parser = OptionParser("usage: %prog [options] arg1 arg2")
parser.add_option("-m", "--model",
                  dest="model",
                  default="",
                  help="model file")
parser.add_option("-w", "--word",
                  dest="word",
                  default="",
                  help="most similar word")
(options, args) = parser.parse_args()

if options.model == "":
    parser.error("model is empty")
if options.word == "":
    parser.error("word is empty")

model = gensim.models.Word2Vec.load(options.model)
print(model.most_similar(options.word))
