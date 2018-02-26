# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class BILSTM_CRF(object):
    def __init__(self, num_classes, hidden_size=128, num_layers=2, batch_size=32,
                 rnn_cell=tf.contrib.rnn.BasicLSTMCell,
                 embedding_matrix=None):
        # Parameter
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.inputs = tf.placeholder(tf.int32, [None, None], name="input")
        self.dropout_prob = tf.placeholder(tf.float16, name="dropout_prob")
        self.targets = tf.placeholder(tf.int32, [None, None], name="targets")
        self.targets_transition = tf.placeholder(tf.int32, [None])
        self.sequence_length = tf.placeholder(shape=[None, ], dtype=tf.int32, name="sequence_length")

        # 需要预训练词向量
        # [batch_size,timestamp_size,hidden_size]
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(embedding_matrix, name="w", dtype=tf.float32)
            self.embedding = tf.nn.embedding_lookup(self.W, self.inputs, name="lookup")

        with tf.name_scope("birnn"):
            fw = tf.nn.rnn_cell.MultiRNNCell(
                [self.rnn(rnn_cell, self.hidden_size, self.dropout_prob)] * self.num_layers)
            bw = tf.nn.rnn_cell.MultiRNNCell(
                [self.rnn(rnn_cell, self.hidden_size, self.dropout_prob)] * self.num_layers)
            # [2, batch_size, sequence_length, hidden_size]
            (output_fw, output_bw), output_states = tf.nn.bidirectional_dynamic_rnn(
                fw,
                bw,
                self.embedding,
                dtype=tf.float32,
                sequence_length=self.sequence_length
            )

        with tf.name_scope("output"):
            # output
            # [batch_size, seq_len, 2 * hidden_size]
            self.outputs = tf.concat([output_fw, output_bw], 2)
            W = tf.Variable(tf.truncated_normal([2 * self.hidden_size, self.num_classes], stddev=0.1))
            b = tf.Variable(tf.zeros([self.num_classes]))
            self.logits = tf.matmul(tf.reshape(self.outputs, [-1, 2 * hidden_size]), W) + b

        with tf.name_scope("crf"):
            self.tags_scores = tf.reshape(self.logits, [self.batch_size, -1, self.num_classes])
            self.sample_length = tf.shape(self.tags_scores)[1]
            self.transitions = tf.get_variable("transitions", [self.num_classes + 1, self.num_classes + 1])

            dummy_val = -1000
            class_pad = tf.Variable(dummy_val * np.ones((self.batch_size, self.sample_length, 1)), dtype=tf.float32)
            self.observations = tf.concat(2, [self.tags_scores, class_pad])

            begin_vec = tf.Variable(np.array([[dummy_val] * self.num_classes + [0] for _ in range(self.batch_size)]),
                                    trainable=False, dtype=tf.float32)
            end_vec = tf.Variable(np.array([[0] + [dummy_val] * self.num_classes for _ in range(self.batch_size)]),
                                  trainable=False, dtype=tf.float32)
            begin_vec = tf.reshape(begin_vec, [self.batch_size, 1, self.num_classes + 1])
            end_vec = tf.reshape(end_vec, [self.batch_size, 1, self.num_classes + 1])

            self.observations = tf.concat(1, [begin_vec, self.observations, end_vec])

            self.mask = tf.cast(tf.reshape(tf.sign(self.targets), [self.batch_size * self.sample_length]), tf.float32)

            # point score
            self.point_score = tf.gather(tf.reshape(self.tags_scores, [-1]),
                                         tf.range(0,
                                                  self.batch_size * self.sample_length) * self.num_classes + tf.reshape(
                                             self.targets, [self.batch_size * self.sample_length]))
            self.point_score *= self.mask

            # transition score
            self.trans_score = tf.gather(tf.reshape(self.transitions, [-1]), self.targets_transition)

            # real score
            self.target_path_score = tf.reduce_sum(self.point_score) + tf.reduce_sum(self.trans_score)

            # all path score
            self.total_path_score, self.max_scores, self.max_scores_pre = self.forward(self.observations,
                                                                                       self.transitions,
                                                                                       self.sample_length)

            # loss
            self.loss = - (self.target_path_score - self.total_path_score)

    def rnn(self, rnn_cell, num_units, output_keep_prob):
        cell = rnn_cell(num_units)
        return tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=output_keep_prob)

    def logsumexp(self, x, axis=None):
        x_max = tf.reduce_max(x, reduction_indices=axis, keep_dims=True)
        x_max_ = tf.reduce_max(x, reduction_indices=axis)
        return x_max_ + tf.log(tf.reduce_sum(tf.exp(x - x_max), reduction_indices=axis))

    def forward(self, observations, transitions, length, is_viterbi=True, return_best_seq=True):
        length = tf.reshape(length, [self.batch_size])
        transitions = tf.reshape(tf.concat(0, [transitions] * self.batch_size), [self.batch_size, 6, 6])
        observations = tf.reshape(observations, [self.batch_size, self.sample_length + 2, 6, 1])
        observations = tf.transpose(observations, [1, 0, 2, 3])
        previous = observations[0, :, :, :]
        max_scores = []
        max_scores_pre = []
        alphas = [previous]
        for t in range(1, self.num_steps + 2):
            previous = tf.reshape(previous, [self.batch_size, 6, 1])
            current = tf.reshape(observations[t, :, :, :], [self.batch_size, 1, 6])
            alpha_t = previous + current + transitions
            if is_viterbi:
                max_scores.append(tf.reduce_max(alpha_t, reduction_indices=1))
                max_scores_pre.append(tf.argmax(alpha_t, dimension=1))
            alpha_t = tf.reshape(self.logsumexp(alpha_t, axis=1), [self.batch_size, 6, 1])
            alphas.append(alpha_t)
            previous = alpha_t

        alphas = tf.reshape(tf.concat(0, alphas), [self.sample_length + 2, self.batch_size, 6, 1])
        alphas = tf.transpose(alphas, [1, 0, 2, 3])
        alphas = tf.reshape(alphas, [self.batch_size * (self.sample_length + 2), 6, 1])

        last_alphas = tf.gather(alphas, tf.range(0, self.batch_size) * (self.sample_length + 2) + length)
        last_alphas = tf.reshape(last_alphas, [self.batch_size, 6, 1])

        max_scores = tf.reshape(tf.concat(0, max_scores), (self.sample_length + 1, self.batch_size, 6))
        max_scores_pre = tf.reshape(tf.concat(0, max_scores_pre), (self.sample_length + 1, self.batch_size, 6))
        max_scores = tf.transpose(max_scores, [1, 0, 2])
        max_scores_pre = tf.transpose(max_scores_pre, [1, 0, 2])

        return tf.reduce_sum(self.logsumexp(last_alphas, axis=1)), max_scores, max_scores_pre
