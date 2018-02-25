# -*- coding: utf-8 -*-


class Tag4(object):
    @staticmethod
    def get_mapping():
        return {"S": 0, "B": 1, "M": 2, "E": 3}

    # S表示单字，B表示词首，M表示词中，E表示词尾
    @staticmethod
    def tag(word):
        if len(word) == 1:
            return [Tag4.get_mapping()["S"]]
        else:
            ret = [Tag4.get_mapping()["B"]]
            for _ in range(len(word[1:-1])):
                ret.append(Tag4.get_mapping()["M"])
            ret.append(Tag4.get_mapping()["E"])
            return ret


class Tag6(object):
    @staticmethod
    def get_mapping():
        return {"S": 0, "B": 1, "C": 2, "D": 3, "M": 4, "E": 5}

    # S表示单字，B表示词首，C表示词中第一个，D表示词中第二个，M表示词中第三个及以后，E表示词尾
    @staticmethod
    def tag(word):
        if len(word) == 1:
            return [Tag6.get_mapping()["S"]]
        elif len(word) == 2:
            return [Tag6.get_mapping()["B"], Tag6.get_mapping()["E"]]
        elif len(word) == 3:
            return [Tag6.get_mapping()["B"], Tag6.get_mapping()["C"], Tag6.get_mapping()["E"]]
        else:
            ret = [Tag6.get_mapping()["B"], Tag6.get_mapping()["C"], Tag6.get_mapping()["D"]]
            for _ in range(len(word[3:-1])):
                ret.append(Tag6.get_mapping()["M"])
            ret.append(Tag6.get_mapping()["E"])
            return ret


if __name__ == "__main__":
    print("tag4")
    print(Tag4.tag("1"))
    print(Tag4.tag("12"))
    print(Tag4.tag("123"))
    print(Tag4.tag("1234"))

    print("tag6")
    print(Tag6.tag("1"))
    print(Tag6.tag("12"))
    print(Tag6.tag("123"))
    print(Tag6.tag("1234"))
    print(Tag6.tag("12345"))
    print(Tag6.tag("123456"))

