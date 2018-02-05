import re
import codecs
import numpy as np


def load_data_and_labels(file, num_class):
    # Load data from files
    examples = list(codecs.open(file, "r", encoding="utf-8").readlines())
    x, y = [], []
    for item in examples:
        data, label = item.split("__label__")
        onehot = int(label.rstrip("\n"))
        value = [0 for _ in range(num_class)]
        value[onehot] = 1
        x.append(data.rstrip("\t"))
        y.append(value)
    return [np.array(x), np.array(y)]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
