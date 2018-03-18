# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class BIRNN_CRF(object):
    def __init__(self, num_classes, max_sequence_length, hidden_size=128, num_layers=2,
                 rnn_cell=tf.nn.rnn_cell.BasicLSTMCell,
                 embedding_matrix=None):
        # Parameter
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.max_sequence_length = max_sequence_length

        self.input_x = tf.placeholder(tf.int32, [None, max_sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, max_sequence_length], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.sequence_lengths = tf.placeholder(shape=[None, ], dtype=tf.int32, name="sequence_lengths")

        # 需要预训练词向量
        # [batch_size,max_sequence_length,hidden_size]
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(embedding_matrix, name="w", dtype=tf.float32)
            self.lookup = tf.nn.embedding_lookup(self.W, self.input_x, name="lookup")

        with tf.name_scope("birnn"):
            fw = tf.nn.rnn_cell.MultiRNNCell(
                [self.rnn(rnn_cell, self.hidden_size, self.dropout_keep_prob)] * self.num_layers)
            bw = tf.nn.rnn_cell.MultiRNNCell(
                [self.rnn(rnn_cell, self.hidden_size, self.dropout_keep_prob)] * self.num_layers)
            # [2, batch_size, max_sequence_length, hidden_size]
            (output_fw, output_bw), output_states = tf.nn.bidirectional_dynamic_rnn(
                fw,
                bw,
                self.lookup,
                dtype=tf.float32,
                sequence_length=self.sequence_lengths
            )
            # [batch_size, max_sequence_length, 2 * hidden_size]
            self.outputs = tf.concat([output_fw, output_bw], 2)

            W = tf.get_variable(
                "W",
                shape=[2 * self.hidden_size, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")

            self.rnn_prediction = tf.matmul(tf.reshape(self.outputs, [-1, 2 * hidden_size]), W) + b
            self.scores = tf.reshape(self.rnn_prediction, [-1, max_sequence_length, self.num_classes])

        with tf.name_scope("crf"):
            # input [batch_size, max_seq_len, num_tags]
            # label [batch_size, max_seq_len]
            # sequence_lengths [batch_size]
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.scores, self.input_y,
                                                                                  self.sequence_lengths)
            self.loss = tf.reduce_mean(-log_likelihood)

            viterbi_sequence, _ = tf.contrib.crf.crf_decode(self.scores, transition_params, self.sequence_lengths)
            self.outputs = tf.identity(viterbi_sequence, name="outputs")

    def rnn(self, rnn_cell, num_units, output_keep_prob):
        cell = rnn_cell(num_units)
        return tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=output_keep_prob)

    @staticmethod
    def viterbi_decode(score, transition_params):
        return tf.contrib.crf.viterbi_decode(score, transition_params)
