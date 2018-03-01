import jieba
import codecs


class WordCut(object):
    def __init__(self, user_dict=None, stop_dict=None):
        self.jieba = jieba
        self.jieba.enable_parallel()
        if user_dict is not None:
            self.user_dict = user_dict
            for user_dict in user_dict.split(","):
                self.jieba.load_userdict(user_dict)
        if stop_dict is not None:
            self.stop_dict = stop_dict
            self.stop_set = set([line.strip() for line in codecs.open(stop_dict, 'r', encoding='utf-8').readlines()])

    def seg(self, sentence):
        sentence = self.jieba.cut(sentence)
        seg = []
        for word in sentence:
            if word not in self.stop_set:
                seg.append(word)
        return seg


if __name__ == "__main__":
    seg = WordCut(user_dict="dict/sougou.dict,dict/sougou-finance.dict,dict/user.dict,dict/token.txt",
                  stop_dict="dict/stopwords.txt")
    print(seg.seg("太阳当空照"))
