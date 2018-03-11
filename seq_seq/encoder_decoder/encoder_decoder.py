# -*- coding: utf-8 -*-

import tensorflow as tf


class EncoderDecoder(object):
    def __init__(self, embeddings, encoder_hidden_size, decoder_hidden_size):
        # pre
        vocab_size = embeddings.shape[0]

        # input data
        self.encoder_inputs = tf.placeholder(shape=[None, None], dtype=tf.int32, name='encoder_inputs')
        self.decoder_inputs = tf.placeholder(shape=[None, None], dtype=tf.int32, name='decoder_inputs')
        self.sequence_length = tf.placeholder(shape=[None, ], dtype=tf.int32, name='sequence_length')

        # embedding
        with tf.device('cpu:0'), tf.name_scope("embedding"):
            self.embeddings = tf.Variable(embeddings, dtype=tf.float32)
            encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.encoder_inputs)
            decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.decoder_inputs)

        with tf.name_scope("encoder"):
            encoder_cell = self.rnn_cell(encoder_hidden_size)
            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell,
                                                                     encoder_inputs_embedded,
                                                                     sequence_length=self.sequence_length,
                                                                     dtype=tf.float32,
                                                                     scope="encoder")
            self.encoder_vector = tf.reshape(encoder_final_state, shape=[1, -1, encoder_hidden_size],
                                             name="encoder_vector")

        with tf.name_scope("decoder"):
            decoder_cell = self.rnn_cell(decoder_hidden_size)
            decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cell,
                                                                     decoder_inputs_embedded,
                                                                     initial_state=encoder_final_state,
                                                                     sequence_length=self.sequence_length,
                                                                     dtype=tf.float32,
                                                                     scope="decoder", )

        with tf.name_scope("output"):
            self.decoder_logits = tf.contrib.layers.fully_connected(decoder_outputs, vocab_size)
            self.decoder_prediction = tf.argmax(self.decoder_logits, 2, output_type=tf.int32, name="decoder_prediction")
            correct = tf.equal(self.decoder_prediction, self.decoder_inputs)
            self.accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")

        with tf.name_scope("loss"):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.one_hot(self.decoder_inputs, depth=vocab_size, dtype=tf.float32),
                logits=self.decoder_logits, )
            self.loss = tf.reduce_mean(self.cross_entropy)

    def rnn_cell(self, hidden_size):
        return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
