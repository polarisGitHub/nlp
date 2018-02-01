# -*- coding: utf-8 -*-
import gensim
import logging
import pipe_line
from optparse import OptionParser
from multiprocessing import cpu_count

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = OptionParser("usage: %prog [options] arg1 arg2")
parser.add_option("-i", "--input",
                  dest="input",
                  default="data/2",
                  help="input file")
parser.add_option("-o", "--old_model",
                  dest="old_model",
                  default="1",
                  help="old model to update")
parser.add_option("-n", "--new_model",
                  dest="new_model",
                  default="2",
                  help="new model to train")
(options, args) = parser.parse_args()

if options.input == "":
    parser.model("input is empty")
if options.old_model == "":
    parser.error("old_model is empty")
if options.new_model == "":
    parser.error("new_model is empty")

word = pipe_line.SplitWord([options.input])

model = gensim.models.Word2Vec.load(options.old_model)
model.build_vocab(word, update=True)
model.train(word, total_examples=model.corpus_count, epochs=model.iter)

model.save(options.new_model)
model.wv.save_word2vec_format(options.new_model + ".txt", binary=False)
