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
