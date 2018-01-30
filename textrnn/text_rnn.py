# -*- coding: utf-8 -*-
# TextRNN: 1. embeddding layer, 2.Bi-RNN layer, 3.FC layer, 5.softmax
import tensorflow as tf


class TextRNN:
    def __init__(self, num_classes, sequence_length, vocab_size, embedding_dim, hidden_size,
                 multi_layer=2, initializer=tf.random_normal_initializer(stddev=0.1)):
        # set hyperparamter
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embedding_dim
        self.hidden_size = hidden_size
        self.multi_layer = multi_layer
        self.initializer = initializer

        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # 1.get emebedding of words in the sentence
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedding = tf.get_variable(
                "embedding",
                shape=[self.vocab_size, self.embed_size],
                initializer=self.initializer)
            self.embedded_words = tf.nn.embedding_lookup(self.embedding, self.input_x)

        # 2.bi-rnn
        with tf.name_scope("bi-rnn"):
            fw_cell = [self.__rnn(self.hidden_size, self.dropout_keep_prob) for _ in range(self.multi_layer)]
            bw_cell = [self.__rnn(self.hidden_size, self.dropout_keep_prob) for _ in range(self.multi_layer)]
            fw = tf.contrib.rnn.MultiRNNCell(fw_cell)
            bw = tf.contrib.rnn.MultiRNNCell(bw_cell)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw, bw, self.embedded_words, dtype=tf.float32)
            output_rnn = tf.concat(outputs, axis=2)  # [batch_size,sequence_length,hidden_size*2]
            self.output_rnn = tf.reduce_mean(output_rnn, axis=1)

        # 3.fc
        with tf.name_scope("output"):
            self.w = tf.get_variable("w", shape=[self.hidden_size * 2, self.num_classes], initializer=self.initializer)
            self.b = tf.get_variable("b", shape=[self.num_classes])
            self.logits = tf.nn.xw_plus_b(self.output_rnn, self.w, self.b, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def __rnn(self, num_units, output_keep_prob):
        cell = tf.contrib.rnn.BasicLSTMCell(num_units)
        return tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=output_keep_prob)
