import codecs
from gensim.models.keyedvectors import KeyedVectors
from utils import tag


class DateIterator(object):
    def __init__(self, file, vocab_path, tag_processor):
        self.buckets = {}
        self.vocab_path = vocab_path
        self.tag_processor = tag_processor

        # 加载词向量
        x, w2v = [], KeyedVectors.load_word2vec_format(self.vocab_path, binary=False)

        # 读取分词文件
        data = list(codecs.open(file, "r", encoding="utf-8").readlines())

        # 获取label
        for item in data:
            sentence = [w2v.vocab[char].index if char in w2v.vocab else w2v.vocab["__UNK__"].index
                        for char in item.strip().replace(" ", "")]
            label = (tag_processor.tag(item.strip()))
            # sentences和labels放入bucket
            bucket_id = len(sentence)
            if bucket_id not in self.buckets:
                self.buckets[bucket_id] = {"sentences": [], "labels": []}
            self.buckets[bucket_id]["sentences"].append(sentence)
            self.buckets[bucket_id]["labels"].append(label)


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

if __name__ == "__main__":
    iter = DateIterator("data/2014_process/word_cut.txt", "data/2014_process/char_cut.w2v.txt", tag.Tag4())
