# -*- coding: utf-8 -*-

# ! /usr/bin/env python

import tensorflow as tf
from gensim.models.keyedvectors import KeyedVectors

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("word2vec_model", "", "word2vec_model which train with gensim")

# Eval Parameters
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("sentence", "", "Checkpoint directory from training run")

FLAGS = tf.flags.FLAGS

w2v = KeyedVectors.load_word2vec_format(FLAGS.word2vec_model, binary=False)
index_word_map, word_index_map = {}, {}
for word, vocab in w2v.vocab.items():
    index_word_map[vocab.index] = word
    word_index_map[word] = vocab.index

sentence_index = [word_index_map[word] if word in word_index_map else word_index_map["__UNK__"].index for word in
                  FLAGS.sentence.split(" ")]


def main(_):
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            encoder_inputs = graph.get_operation_by_name("encoder_inputs").outputs[0]
            decoder_inputs = graph.get_operation_by_name("decoder_inputs").outputs[0]
            sequence_length = graph.get_operation_by_name("sequence_length").outputs[0]
            encoder_vector = graph.get_operation_by_name("encoder/encoder_vector").outputs[0]
            decoder_prediction = graph.get_operation_by_name("output/decoder_prediction").outputs[0]
            accuracy = graph.get_operation_by_name("output/accuracy").outputs[0]

            # Tensors we want to evaluate

            # Collect the predictions here
            encoder, prediction, a = sess.run([encoder_vector, decoder_prediction, accuracy],
                                              feed_dict={encoder_inputs: [sentence_index],
                                                         decoder_inputs: [sentence_index],
                                                         sequence_length: [len(sentence_index)]})
            print(sentence_index, prediction, a)
            prediction_sentence = ""
            for index in prediction[0]:
                prediction_sentence += index_word_map[index]
            print(prediction_sentence)


if __name__ == "__main__":
    tf.app.run()
