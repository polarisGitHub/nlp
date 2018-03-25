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
        self.bucket_train_data, self.bucket_test_data = {}, {}

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

        # 对内存友好
        max_buckets_id = int(max_sequence_length / 10)
        buckets = {i: {
            "sentences": [],
            "labels": [],
            "lengths": []
        } for i in range(1, max_buckets_id + 1)}

        unk = w2v.vocab.get("_unk_")
        for item in data:
            item = item.strip()
            sentence = [w2v.vocab.get(char, unk).index for char in item.replace(" ", "")]
            label = tag_processor.tag(item)

            item_len = len(sentence)
            bucket_id = min(int(item_len / 10) + 1, max_buckets_id)

            buckets[bucket_id]["sentences"].append(self.padding(sentence, bucket_id * 10))
            buckets[bucket_id]["labels"].append(self.padding(label, bucket_id * 10))
            buckets[bucket_id]["lengths"].append(min(len(sentence), max_sequence_length))

        print("tag data done")

        # 划分训练集合测试集
        np.random.seed(42)
        for bucket_id, bucket in buckets.items():
            convert_data = np.array(list(zip(bucket["sentences"], bucket["labels"], bucket["lengths"])))
            shuffle_indices = np.random.permutation(np.arange(len(convert_data)))
            shuffled = convert_data[shuffle_indices]
            dev_sample_index = int(train_data_ratio * float(len(bucket["sentences"])))
            self.bucket_train_data[bucket_id] = shuffled[:dev_sample_index]
            self.bucket_test_data[bucket_id] = shuffled[dev_sample_index:]

    def get_train_data(self):
        return self.bucket_train_data

    def get_dev_data(self):
        return self.bucket_test_data

    def expand_padding_batch(self, data, max_sequence_len):
        padding_sentences, padding_labels, real_lengths = [], [], []
        sentences, labels, lengths = zip(*data)
        for i in range(len(sentences)):
            padding_sentences.append(self.padding(sentences[i], max_sequence_len))
            padding_labels.append(self.padding(labels[i], max_sequence_len))
            real_lengths.append(lengths[i])
        return padding_sentences, padding_labels, real_lengths

    def expand_buckets(self, data):
        padding_sentences, padding_labels, real_lengths = [], [], []
        for _, bucket in data.items():
            sentences, labels, lengths = zip(*bucket)
            for i in range(len(sentences)):
                padding_sentences.append(sentences[i])
                padding_labels.append(labels[i])
                real_lengths.append(lengths[i])
        return padding_sentences, padding_labels, real_lengths

    def padding_batch(self, data, max_len):
        padding = []
        for item in data:
            padding.append(self.padding(item, max_len))
        return padding

    def batch_iter(self, data, batch_size=64, num_epochs=100):
        for epoch in range(num_epochs):
            for bucket_id, bucket in data.items():
                bucket_size = len(bucket)
                num_batches_per_epoch = int((bucket_size - 1) / batch_size) + 1
                shuffle_indices = np.random.permutation(np.arange(bucket_size))
                shuffled_data = bucket[shuffle_indices]
                print(bucket_id, bucket_size, num_batches_per_epoch)
                for batch_num in range(num_batches_per_epoch):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, bucket_size)
                    yield shuffled_data[start_index:end_index]

    def padding(self, origin_list, max_sequence_len):
        sentence_len = len(origin_list)
        if sentence_len > max_sequence_len:
            return origin_list[0:max_sequence_len]
        return origin_list + ([self._padding_token] * (max_sequence_len - sentence_len))
