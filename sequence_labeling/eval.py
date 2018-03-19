#! /usr/bin/env python

import tensorflow as tf

from utils import tag
from birnn_crf import BIRNN_CRF
from data_helpers import DateIterator

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_file", "data/2014_process/word_cut.txt", "Data source for the train.")
tf.flags.DEFINE_string("w2v_model", "w2v/char_cut.w2v.txt", "word2vec_model which train with gensim")
tf.flags.DEFINE_string("tag", "tag4", "use tag4 or tag6")

# Eval Parameters

tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1521393606/checkpoints", "Checkpoint directory from training run")

FLAGS = tf.flags.FLAGS

data = ["斯宾塞、“冰箱”弗里奇、贝瑟尼和玛莎是某高中的问题学生"]
if FLAGS.tag == "tag4":
    data_iter = DateIterator(data=data, vocab_path=FLAGS.w2v_model, tag_processor=tag.Tag4(), )
elif FLAGS.tag == "tag6":
    data_iter = DateIterator(data=data, vocab_path=FLAGS.w2v_model, tag_processor=tag.Tag6(), )
else:
    raise ValueError(FLAGS.tag + "not support")


def main(_):
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            inpux_x = graph.get_operation_by_name("input_x").outputs[0]
            sequence_length = graph.get_operation_by_name("sequence_lengths").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            outputs = graph.get_operation_by_name("crf/outputs").outputs[0]

            # Collect the predictions here
            batches = data_iter.batch_iter(batch_size=1, num_epochs=1)
            for batch in batches:
                batch_data, batch_labels, batch_lengths = zip(*batch)
                nn_outputs = sess.run(outputs, feed_dict={inpux_x: batch_data,
                                                          sequence_length: batch_lengths,
                                                          dropout_keep_prob: 1.0})
                for i in range(len(nn_outputs)):
                    cut = ""
                    for j in range(batch_lengths[i]):
                        tag = nn_outputs[i][j]
                        if tag == 0 or tag == 1:
                            cut += " "
                        cut += data[i][j]
                    print(cut.strip())


if __name__ == "__main__":
    tf.app.run()
