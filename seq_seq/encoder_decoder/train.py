# -*- coding: utf-8 -*-

# 无监督学习，input和output相同，目的是获取rnn state用于比较句子相似性
import os
import time
import datetime
import numpy as np
import tensorflow as tf
from gensim.models.keyedvectors import KeyedVectors

import data_helpers
from encoder_decoder import EncoderDecoder

print('TensorFlow Version: {}'.format(tf.__version__))

tf.flags.DEFINE_string("data_file", "data/sentence.csv", "Data source for the train.")
tf.flags.DEFINE_string("word2vec_model", "w2v/corpus.w2v.txt", "word2vec_model which train by gensim")

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("checkpoint_every", 10, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 5)")

FLAGS = tf.flags.FLAGS


def main(_):
    # loading
    print("Loading data...")
    data_list = data_helpers.load_data_and_labels(FLAGS.data_file)

    print("Loading w2v")
    sentence, sentence_len, = [], []
    w2v = KeyedVectors.load_word2vec_format(FLAGS.word2vec_model, binary=False)
    vocab, embeddings = w2v.vocab, np.zeros((len(w2v.index2word), w2v.vector_size), dtype=np.float32)

    print("convert sentence to index")
    for k, v in vocab.items():
        embeddings[v.index] = w2v[k]

    max_len = -1
    for item in data_list:
        sentence_index = [w2v.vocab[word].index if word in w2v.vocab else w2v.vocab["__UNK__"].index
                          for word in item.split(" ")]
        sentence.append(sentence_index)
        length = len(sentence_index)
        sentence_len.append(length)
        if length > max_len:
            max_len = length
    # 补padding，不然数据feed不进去
    for item in sentence:
        item.extend([0] * (max_len - len(item)))
    print("Vocabulary Size: {:d}".format(len(w2v.vocab)))

    # save path
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs/enc_dec/", timestamp))
    print("Writing to {}\n".format(out_dir))

    # checkpoint
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # build graph
    with tf.Graph().as_default():
        sess = tf.InteractiveSession()

        enc_dec = EncoderDecoder(
            embeddings=embeddings,
            encoder_hidden_size=64,
            decoder_hidden_size=64, )
        # train op
        global_step, optimizer = tf.Variable(0, name="global_step", trainable=False), tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(enc_dec.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Summaries
        loss_summary = tf.summary.scalar("loss", enc_dec.loss)
        acc_summary = tf.summary.scalar("accuracy", enc_dec.accuracy)

        summary_op, summary_dir = tf.summary.merge([loss_summary, acc_summary]), os.path.join(out_dir, "summaries")
        summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

        # saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints, save_relative_paths=True)

        # init
        sess.run(tf.global_variables_initializer())

        def train_step(sentence, sentence_len):
            """
            A single training step
            """
            feed_dict = {
                enc_dec.encoder_inputs: sentence,
                enc_dec.decoder_inputs: sentence,
                enc_dec.sequence_length: sentence_len
            }
            _, step, summaries, loss, acc = sess.run(
                [train_op, global_step, summary_op, enc_dec.loss, enc_dec.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))
            summary_writer.add_summary(summaries, step)

        # do train
        batches = data_helpers.batch_iter(list(zip(sentence, sentence_len)), FLAGS.batch_size, FLAGS.num_epochs)
        for batch in batches:
            train_sentence, train_sentence_len = zip(*batch)
            train_step(list(train_sentence), list(train_sentence_len))
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


if __name__ == "__main__":
    tf.app.run()
