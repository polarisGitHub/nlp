import codecs
import numpy as np
from gensim.models.keyedvectors import KeyedVectors


class DateIterator(object):
    def __init__(self, file=None, data=None, vocab_path=None, tag_processor=None, max_sequence_length=100,
                 train_data_ratio=0.9):
        self.vocab_path = vocab_path
        self.tag_processor = tag_processor
        self.max_sequence_length = max_sequence_length

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

        # 获取转换数据
        padding_sentences, padding_labels, length = [], [], []
        for item in data:
            sentence = [w2v.vocab[char].index if char in w2v.vocab else w2v.vocab["__UNK__"].index
                        for char in item.strip().replace(" ", "")]
            label = tag_processor.tag(item.strip())

            padding_sentences.append(self.padding(sentence, 0, max_sequence_length))
            padding_labels.append(self.padding(label, 0, max_sequence_length))
            length.append(len(sentence) if len(sentence) < max_sequence_length else max_sequence_length)

        convert_data = np.array(list(zip(padding_sentences, padding_labels, length)))

        # 划分训练集合测试集
        np.random.seed()
        convert_data = np.array(convert_data)
        shuffle_indices = np.random.permutation(np.arange(len(convert_data)))
        shuffled = convert_data[shuffle_indices]

        dev_sample_index = int(train_data_ratio * float(len(data)))
        self.train_data, self.dev_data = shuffled[:dev_sample_index], shuffled[dev_sample_index:]

        del data, convert_data, shuffle_indices

    def get_train_data(self):
        return self.train_data

    def get_dev_data(self):
        return self.dev_data

    def batch_iter(self, data, batch_size=64, num_epochs=100):
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        for epoch in range(num_epochs):
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    def padding(self, origin_list, padding_token, max_sequence_len):
        sentence_len = len(origin_list)
        if sentence_len > max_sequence_len:
            return origin_list[0:max_sequence_len]
        return origin_list + ([padding_token] * (max_sequence_len - sentence_len))
