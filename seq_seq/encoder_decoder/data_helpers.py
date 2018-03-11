import re
import codecs
import numpy as np
import itertools
from collections import Counter


def load_data_and_labels(file):
    # Load data from files
    examples = list(codecs.open(file, "r", encoding="utf-8").readlines())
    x, len = [], []
    for item in examples:
        x.append(item.strip())
    return x


def batch_iter(data, batch_size, num_epochs):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
