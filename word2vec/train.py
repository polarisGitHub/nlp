# -*- coding: utf-8 -*-

# https://rare-technologies.com/word2vec-tutorial/
import gensim
import logging
import Pipeline

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
iter = Pipeline.Character(["data/extract_wiki"])

model = gensim.models.Word2Vec(iter, sg=1, size=128, workers=4)
model.save('data/model')