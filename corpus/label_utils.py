import codecs


class LabelUtils(object):
    @staticmethod
    def build_label_index(input_file, output_file):
        labels = []
        with codecs.open(input_file, "r", encoding="utf-8") as label:
            for line in label:
                labels.append(line.strip())

        with codecs.open(output_file, "w", encoding="utf-8") as writter:
            for index, label in enumerate(labels):
                writter.write(label + "\t__label__" + str(index) + "\n")

    @staticmethod
    def build_label_map(label_file):
        label_index_map = {}
        with codecs.open(label_file, "r", encoding="utf-8") as line:
            for l in line:
                label, index = l.split("__label__")
                label_index_map[label.strip()] = int(index.strip())
        return label_index_map


if __name__ == "__main__":
    LabelUtils.build_label_index(input_file="data/label/20180130_14_26_31_category.csv",
                                 output_file="data/label/label_index.csv")
    label_map = LabelUtils.build_label_map("data/label/label_index.csv")
    print(label_map)
