# -*- coding: utf-8 -*-

import os
import os.path


def find_files(root):
    files = []
    for file in os.listdir(root):
        path = os.path.join(root, file)
        if os.path.isdir(path):
            files.extend(find_files(path))
        else:
            files.append(path)
    return files


if __name__ == "__main__":
    txt = find_files("../data/2014")
    print(len(txt), txt[0])
