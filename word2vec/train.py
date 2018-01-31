# -*- coding: utf-8 -*-

# https://rare-technologies.com/word2vec-tutorial/
import gensim
import logging
import pipe_line
from optparse import OptionParser
from multiprocessing import cpu_count

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = OptionParser("usage: %prog [options] arg1 arg2")
parser.add_option("-i", "--input",
                  dest="input",
                  default="",
                  help="input file")
parser.add_option("-m", "--model",
                  dest="model",
                  default="",
                  help="save model")
(options, args) = parser.parse_args()

if options.input == "":
    parser.model("input is empty")
if options.model == "":
    parser.error("model is empty")

word = pipe_line.SplitWord([options.input])

# 输入是带__UNK__的，min_count设为1
model = gensim.models.Word2Vec(word, min_count=1, sg=1, hs=1, size=128, workers=cpu_count())
model.save(options.model)
model.wv.save_word2vec_format(options.model + ".txt", binary=False)
