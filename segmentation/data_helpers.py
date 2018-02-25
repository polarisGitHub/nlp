import codecs
import numpy as np


class DateIterator(object):
    def __init__(self, file, tag):
        self.buckets = {}
        data = list(codecs.open(file, "r", encoding="utf-8").readlines())
        labels = []
        for sentence in data:
            labels.append(tag.tag(sentence))
        bucket_id = len(data[0].split(" "))
        self.buckets = {bucket_id: {"data": data, "labels": labels}}

    # def batch_iter(self, buckets, batch_size, num_epochs):
    #     """
    #     Generates a batch iterator for a dataset.
    #     """
    #     data = np.array(data)
    #     data_size = len(data)
    #     num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    #     for epoch in range(num_epochs):
    #         shuffle_indices = np.random.permutation(np.arange(data_size))
    #         shuffled_data = data[shuffle_indices]
    #         for batch_num in range(num_batches_per_epoch):
    #             start_index = batch_num * batch_size
    #             end_index = min((batch_num + 1) * batch_size, data_size)
    #             yield shuffled_data[start_index:end_index]
