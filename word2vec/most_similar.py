# -*- coding: utf-8 -*-

from gensim.models.keyedvectors import KeyedVectors
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

word_vectors = KeyedVectors.load_word2vec_format(options.model, binary=False)
print(word_vectors.most_similar(options.word))
