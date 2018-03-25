# -*- coding: utf-8 -*-


class Tag4(object):
    def get_mapping(self):
        return {"S": 0, "B": 1, "M": 2, "E": 3}

    # S表示单字，B表示词首，M表示词中，E表示词尾
    def tag(self, sentence):
        ret = []
        for word in sentence.strip().split(" "):
            ret.extend(self.__do_tag(word))
        return ret

    def __do_tag(self, word):
        word_len = len(word)
        if len(word) == 1:
            return [self.get_mapping()["S"]]
        else:
            ret = [self.get_mapping()["B"]]
            ret.extend([self.get_mapping()["M"]] * (word_len - 2))
            ret.append(self.get_mapping()["E"])
            return ret


class Tag6(object):
    def get_mapping(self):
        return {"S": 0, "B": 1, "C": 2, "D": 3, "M": 4, "E": 5}

    def tag(self, sentence):
        ret = []
        for word in sentence.strip().split(" "):
            ret.extend(self.__do_tag(word))
        return ret

    # S表示单字，B表示词首，C表示词中第一个，D表示词中第二个，M表示词中第三个及以后，E表示词尾
    def __do_tag(self, word):
        word_len = len(word)
        if word_len == 1:
            return [self.get_mapping()["S"]]
        elif word_len == 2:
            return [self.get_mapping()["B"], self.get_mapping()["E"]]
        elif word_len == 3:
            return [self.get_mapping()["B"], self.get_mapping()["C"], self.get_mapping()["E"]]
        else:
            ret = [self.get_mapping()["B"], self.get_mapping()["C"], self.get_mapping()["D"]]
            ret.extend([self.get_mapping()["M"]] * (word_len - 4))
            ret.append(self.get_mapping()["E"])
            return ret


if __name__ == "__main__":
    print("tag4")
    print(Tag4().tag("1 12 123 1234"))

    print("tag6")
    print(Tag6().tag("1 12 123 1234 12345 123456"))
