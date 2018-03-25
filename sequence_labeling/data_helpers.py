import codecs
import numpy as np
from gensim.models.keyedvectors import KeyedVectors


class DateIterator(object):
    _padding_token = 0

    def __init__(self, file=None, data=None, vocab_path=None, tag_processor=None, max_sequence_length=100,
                 train_data_ratio=0.9):
        self.vocab_path = vocab_path
        self.tag_processor = tag_processor
        self.max_sequence_length = max_sequence_length
        self.train_data, self.test_data = [], []

        # 加载词向量
        x, w2v = [], KeyedVectors.load_word2vec_format(self.vocab_path, binary=False)
        self.embedding_matrix = np.zeros((len(w2v.index2word), w2v.vector_size), dtype=np.float32)

        for k, v in w2v.vocab.items():
            self.embedding_matrix[v.index] = w2v[k]

        # 获取分词数据
        if file is not None:
            data = list(codecs.open(file, "r", encoding="utf-8").readlines())

        if data is None:
            raise ValueError("data is None")

        print("read data done")

        # 对内存友好，不做padding
        sentences, labels, lengths = [], [], []
        unk = w2v.vocab.get("_unk_")
        for item in data:
            item = item.strip()
            sentence = [w2v.vocab.get(char, unk).index for char in item.replace(" ", "")]
            label = tag_processor.tag(item)

            sentences.append(sentence)
            labels.append(label)
            lengths.append(min(len(sentence), self.max_sequence_length))

        print("tag data done")

        # 划分训练集合测试集
        np.random.seed(42)
        convert_data = np.array(list(zip(sentences, labels, lengths)))
        shuffle_indices = np.random.permutation(np.arange(len(convert_data)))
        shuffled = convert_data[shuffle_indices]
        dev_sample_index = int(train_data_ratio * float(len(convert_data)))
        self.train_data = shuffled[:dev_sample_index]
        self.test_data = shuffled[dev_sample_index:]

    def get_train_data(self):
        return self.train_data

    def get_dev_data(self):
        return self.test_data

    def padding_batch(self, data, max_sequence_len):
        padding = []
        for item in data:
            padding.append(self.padding(item, max_sequence_len))
        return padding

    def batch_iter(self, data, batch_size=64, num_epochs=100):
        for epoch in range(num_epochs):
            data_size = len(data)
            num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    def padding(self, origin_list, max_sequence_len):
        sentence_len = len(origin_list)
        if sentence_len > max_sequence_len:
            return origin_list[0:max_sequence_len]
        return origin_list + ([self._padding_token] * (max_sequence_len - sentence_len))
