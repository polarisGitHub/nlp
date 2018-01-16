# -*- coding: utf-8 -*-

import codecs
from utils import full_half_converter

# http://www.sogou.com/labs/resource/ca.php
# http://www.sogou.com/labs/resource/cs.php
files = ["./data/news_sohusite_xml.dat", "./data/news_tensite_xml.dat"]

with codecs.open("./data/sougo", "w", encoding="utf-8") as extracted:
    for file in files:
        with codecs.open(file, "r", encoding="GB18030") as reader:
            while True:
                line = reader.readline()
                if line == "":
                    break
                if line.startswith("<content>"):
                    line = line.replace("", " ")
                    line = line.strip("\n")
                    line = line.replace("<content>", "")
                    line = line.replace("</content>", "")
                    if len(line) != 0:
                        line = full_half_converter.convert(line,
                                                           full_half_converter.FH_ALPHA,
                                                           full_half_converter.FH_NUM,
                                                           full_half_converter.FH_PUNCTUATION,
                                                           full_half_converter.FH_SPACE)
                        extracted.write(line)
                        extracted.write("\n")
