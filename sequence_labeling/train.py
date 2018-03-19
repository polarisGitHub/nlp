#! /usr/bin/env python

import os
import time
import datetime
import tensorflow as tf

from utils import tag
from birnn_crf import BIRNN_CRF
from data_helpers import DateIterator

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_file", "data/2014_process/word_cut.txt", "Data source for the train.")
tf.flags.DEFINE_string("w2v_model", "w2v/char_cut.w2v.txt", "word2vec_model which train with gensim")
tf.flags.DEFINE_string("tag", "tag4", "use tag4 or tag6")

# Model Hyperparameters
tf.flags.DEFINE_integer("max_sequence_length", 100, "Data source for the train.")
tf.flags.DEFINE_integer("hidden_size", 128, "rnn hidden size (default: 128)")
tf.flags.DEFINE_integer("num_layers", 2, "multi rnn layer (default: '2')")
tf.flags.DEFINE_string("rnn_cell", "lstm", "lstm or gur")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 20)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def main(_):
    # Data Preparation
    # ==================================================

    # Load data
    if FLAGS.tag == "tag4":
        data_iter = DateIterator(file=FLAGS.train_file, vocab_path=FLAGS.w2v_model, tag_processor=tag.Tag4(),
                                 max_sequence_length=FLAGS.max_sequence_length)
        num_classes = 4
    elif FLAGS.tag == "tag6":
        data_iter = DateIterator(file=FLAGS.train_file, vocab_path=FLAGS.w2v_model, tag_processor=tag.Tag6(),
                                 max_sequence_length=FLAGS.max_sequence_length)
        num_classes = 6
    else:
        raise ValueError(FLAGS.tag + "not support")

    if FLAGS.rnn_cell == "lstm":
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell
    elif FLAGS.rnn_cell == "gru":
        rnn_cell = tf.nn.rnn_cell.GRUCell
    else:
        raise ValueError(FLAGS.rnn_cell + "not support")

    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            rnn = BIRNN_CRF(
                num_classes=num_classes,
                max_sequence_length=FLAGS.max_sequence_length,
                hidden_size=FLAGS.hidden_size,
                num_layers=FLAGS.num_layers,
                rnn_cell=rnn_cell,
                embedding_matrix=data_iter.embedding_matrix
            )

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(rnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", rnn.loss)
            acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints, save_relative_paths=True)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch, lengths_batch):
                """
                A single training step
                """
                feed_dict = {
                    rnn.input_x: x_batch,
                    rnn.input_y: y_batch,
                    rnn.sequence_lengths: lengths_batch,
                    rnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, acc = sess.run(
                    [train_op, global_step, train_summary_op, rnn.loss, rnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, lengths_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    rnn.input_x: x_batch,
                    rnn.input_y: y_batch,
                    rnn.sequence_lengths: lengths_batch,
                    rnn.dropout_keep_prob: 1.0
                }

                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, rnn.loss, rnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            batches = data_iter.batch_iter(data_iter.get_train_data(), batch_size=FLAGS.batch_size,
                                           num_epochs=FLAGS.num_epochs)
            dev_data, dev_label, dev_lengths = zip(*data_iter.get_dev_data())
            for batch in batches:
                batch_data, batch_labels, batch_lengths = zip(*batch)
                # train
                train_step(batch_data, batch_labels, batch_lengths)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(dev_data, dev_label, dev_lengths, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


if __name__ == "__main__":
    tf.app.run()
