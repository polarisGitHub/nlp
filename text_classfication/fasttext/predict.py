# -*- coding: utf-8 -*-
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import fasttext

classifier = fasttext.load_model('model/model.bin', label_prefix='__label__')
result = classifier.predict_proba(["银行卡 找 不到"], k=3)
print(result)
