# -*- coding: utf-8 -*-


class Tag4(object):
    @staticmethod
    def get_mapping():
        return {"S": 0, "B": 1, "M": 2, "E": 3}

    # S表示单字，B表示词首，M表示词中，E表示词尾
    @staticmethod
    def tag(word):
        if not isinstance(word, str):
            return ""
        if len(word) == 0:
            return ""
        elif len(word) == 1:
            return "S"
        else:
            ret = "B"
            for _ in range(len(word[1:-1])):
                ret += "M"
            ret += "E"
            return ret


if __name__ == "__main__":
    print(Tag4.tag("1"))
    print(Tag4.tag("12"))
    print(Tag4.tag("123"))
    print(Tag4.tag("1234"))
    print(Tag4.tag("12345"))
