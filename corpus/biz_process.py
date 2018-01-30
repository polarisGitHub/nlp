import codecs
import pandas as pd
from label_utils import LabelUtils
from word_cut import WordCut

# 需要处理的文件
origins = ["train.csv", "test.csv", "similar.csv", "qcategory.csv"]

label_origin = "data/biz/label/category.csv"
label_process = "data/biz/label/label_index.csv"

LabelUtils.build_label_index(input_file=label_origin, output_file=label_process)
label_index_map = LabelUtils.build_label_map(label_process)
index_label_map = zip(label_index_map.values(), label_index_map.keys())

word_cut = WordCut(user_dict="dict/sougou.dict,dict/sougou-finance.dict,dict/user.dict",
                   stop_dict="dict/stopwords.txt")
for origin in origins:
    print(origin)
    process_sentence = []
    process_label = []
    reader = pd.read_csv("data/biz/origin/" + origin, header=None)
    for index, row in reader.iterrows():
        sentence, label = row[0].strip(), row[1].strip()
        if label in label_index_map:
            process_sentence.append(" ".join(word_cut.seg(sentence)))
            process_label.append("__label__" + str(label_index_map[label]))
    predictions = pd.concat([pd.Series(process_sentence), pd.Series(process_label)], axis=1)
    predictions.to_csv("data/biz/process/" + origin, index=False, sep='\t', encoding="utf-8", header=False)
