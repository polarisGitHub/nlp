import codecs

# read ori data
sentence = []
true_label = []
with codecs.open("data/2018_01_02_all_train_data_test.csv", "r", encoding="utf-8") as test_data:
    for line in test_data:
        s, l = line.split("__label__")
        sentence.append(s.strip())
        true_label.append(int(l.strip()))

# read label mapping
label_mapping = {}
with codecs.open("data/2018_01_02_label.csv", "r", encoding="utf-8") as fasttext:
    for line in fasttext:
        label, description = line.split(",")
        label_mapping[int(label.split("__label__")[1].strip())] = description.strip()

# read fasttext
fasttext_predict = []
with codecs.open("data/2018-01-02_predict.csv", "r", encoding="utf-8") as fasttext:
    for line in fasttext:
        fasttext_predict.append(int(line.split("__label__")[1].strip()))

# read textcnn
textcnn_predict = []
with codecs.open("data/prediction.csv", "r", encoding="utf-8") as fasttext:
    for line in fasttext:
        s, l = line.split(",")
        textcnn_predict.append(int(float(l.strip())))
# find same error
same_error = []
for index, s in enumerate(sentence):
    if true_label[index] != fasttext_predict[index] and fasttext_predict[index] == textcnn_predict[index]:
        same_error.append(
            sentence[index] + "," + label_mapping[true_label[index]] + " ->" + label_mapping[fasttext_predict[index]])

with codecs.open("data/same_error.csv", "w", encoding="utf-8") as writter:
    for error in same_error:
        writter.write(error + "\n")
