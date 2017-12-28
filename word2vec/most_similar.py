# -*- coding: utf-8 -*-

import gensim

model = gensim.models.Word2Vec.load("data/model")
print(model.most_similar("1"))
