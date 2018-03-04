import codecs
import numpy as np
from gensim.models.keyedvectors import KeyedVectors


class DateIterator(object):
    def __init__(self, file, vocab_path, tag_processor):
        self.buckets = {}  # 随机后的numpy数据
        self.buckets_raw = {}  # 原始数据list
        self.vocab_path = vocab_path
        self.tag_processor = tag_processor

        # 加载词向量
        x, w2v = [], KeyedVectors.load_word2vec_format(self.vocab_path, binary=False)
        self.embedding_matrix = np.zeros((len(w2v.index2word), w2v.vector_size), dtype=np.float32)

        for k, v in w2v.vocab.items():
            self.embedding_matrix[v.index] = w2v[k]

        # 读取分词文件
        data = list(codecs.open(file, "r", encoding="utf-8").readlines())

        # 获取label
        for item in data:
            sentence = [w2v.vocab[char].index if char in w2v.vocab else w2v.vocab["__UNK__"].index
                        for char in item.strip().replace(" ", "")]
            label = (tag_processor.tag(item.strip()))
            # sentences和labels放入bucket
            bucket_id = len(sentence)
            if bucket_id not in self.buckets_raw:
                self.buckets_raw[bucket_id] = {"sentences": [], "labels": []}
            self.buckets_raw[bucket_id]["sentences"].append(sentence)
            self.buckets_raw[bucket_id]["labels"].append(label)

    def batch_iter(self, batch_size, num_epochs):
        """
        Generates a batch iterator for a dataset.
        """
        for _ in range(num_epochs):
            self.__init_epochs()  # 每个epochs开始前初始化数据
            while self.__has_bucket_epochs(batch_size):  # 是否一个epochs执行完了，每个bucket都执行过了
                bucket_id = self.__find_bucket_id()  # 随机一个bucket_id
                batch_num = self.buckets[bucket_id]["bucket_batch_num"]
                bucket_size = self.buckets[bucket_id]["bucket_size"]

                self.buckets[bucket_id]["bucket_batch_num"] = batch_num + 1

                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, bucket_size)

                sentences, labels = self.buckets[bucket_id]["sentences"], self.buckets[bucket_id]["labels"]
                yield sentences[start_index:end_index], labels[start_index:end_index], [bucket_id] * batch_size

    def __has_bucket_epochs(self, batch_size):
        for bucket_id in list(self.buckets.keys()):
            batch_num = self.buckets[bucket_id]["bucket_batch_num"]
            bucket_size = self.buckets[bucket_id]["bucket_size"]
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, bucket_size)
            if end_index - start_index < batch_size:  # 模型不允许不完整的batch
                self.buckets.pop(bucket_id)
        return len(self.buckets) > 0

    def __init_epochs(self):
        for bucket_id, bucket in self.buckets_raw.items():
            bucket_size = len(bucket["sentences"])
            self.buckets[bucket_id] = {
                "sentences": np.array(self.buckets_raw[bucket_id]["sentences"]),
                "labels": np.array(self.buckets_raw[bucket_id]["labels"])
            }
            # 打乱顺序
            shuffle_indices = np.random.permutation(np.arange(bucket_size))
            self.buckets[bucket_id]["sentences"] = self.buckets[bucket_id]["sentences"][shuffle_indices]
            self.buckets[bucket_id]["labels"] = self.buckets[bucket_id]["labels"][shuffle_indices]
            # 初始化bucket参数
            self.buckets[bucket_id]["bucket_size"] = bucket_size
            self.buckets[bucket_id]["bucket_batch_num"] = 0

    def __find_bucket_id(self):
        bucket_ids = list(self.buckets.keys())
        index = np.random.permutation(np.arange(len(bucket_ids)))[0]
        return bucket_ids[index]


if __name__ == "__main__":
    from utils import tag

    batch_iter = DateIterator("data/2014_process/word_cut.txt", "w2v/char_cut.w2v.txt", tag.Tag4()).batch_iter(
        batch_size=1, num_epochs=1)
    next(batch_iter)
