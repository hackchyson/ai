import argparse  # Command line parsing
import configparser  # Saving the models parameters
import datetime  # Chronometer
import os  # Files management
import tensorflow as tf
import numpy as np
import math
from tqdm import tqdm  # Progress bar
from tensorflow.python import debug as tf_debug
from model import Model
from data import Data
from pandas import Series

import config as chat_config


class TrainModel:
    def __init__(self, model_tag):
        self.config = chat_config.Config()
        self.data = Data()
        self.data.load()
        self.model = Model(len(self.data.word2id), self.data.start_token)
        # summary and saver
        self.writer = tf.summary.FileWriter(os.path.join(self.config.model_dir, model_tag))
        self.saver = tf.train.Saver()
        # init
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.sess.run(tf.global_variables_initializer())
        self.global_step = 0
        print('Begin training...')
        self.model_name = os.path.join(self.config.model_dir, model_tag)
        if os.listdir(self.config.model_dir):
            # if os.path.exists(self.model_name):
            print(self.model_name)
            if False:

                self.saver.restore(self.sess, self.model_name)
                self.train(self.sess, False)
            else:
                print('No previous model found, staring from clean directory: {}'.format(self.config.model_dir))
                self.train(self.sess, True)

    def train(self, sess, train_from_scratch):
        merged_summaries = tf.summary.merge_all()

        if train_from_scratch:
            print('No previous model found, starting from clean directory: {}'.format(self.config.model_dir))
            self.writer.add_graph(sess.graph)  # First time only
        try:
            for e in range(self.config.num_epochs):
                # print('----- epoch {}/{} ; (lr=[]) -----'.format(e + 1, self.config.num_epochs,
                #                                                  self.config.learning_rate))
                batches = self.data.get_batch()
                start = datetime.datetime.now()
                for b in tqdm(batches, desc="Training"):
                    ops, feed_dict = self.model.step(b)
                    assert len(ops) == 2
                    _, loss, summary = self.sess.run([ops, merged_summaries], feed_dict=feed_dict)
                    self.writer.add_summary(summary, self.global_step)
                    self.global_step += 1
                    if self.global_step % 10 == 0:
                        perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                        print("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (self.global_step, loss, perplexity))
                        print("----- Epoch {}/{} ; (lr={}) -----".format(e + 1, self.config.num_epochs,
                                                                         self.config.learning_rate))
                        self.saver.save(self.sess, self.model_name)
                        print('Model saved.')
                end = datetime.datetime.now()
                print('Epoch finished in {}'.format(end - start))
        except (KeyboardInterrupt, SystemExit):
            print('Interruption detected, exiting the program...')
        self.saver.save(self.sess, self.model_name)
        print('Model saved.')


if __name__ == "__main__":
    train = TrainModel('hack-test-01')
