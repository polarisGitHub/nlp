# _*_coding:utf-8 _*_
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import fasttext

classifier = fasttext.supervised("data/train.csv", "model/model", label_prefix="__label__", epoch=60)
