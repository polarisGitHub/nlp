# -*- coding: utf-8 -*-

import gensim

model = gensim.models.Word2Vec.load("data/corpus.model")
print(model.most_similar("unk"))
