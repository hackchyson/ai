"""
Model to predict the next sentence given an input sequence

"""
import tensorflow as tf
from data import Batch
from config import Config


class Model:
    """
    Implementation of a seq2seq model.
    Architecture:
        Encoder/decoder
        2 LTSM layers
    """

    def __init__(self, vocabulary_size, start_token):

        print("Model creation...")
        self.conf = Config()
        self.dtype = tf.float32
        # Placeholders
        self.encoder_inputs = None
        self.decoder_inputs = None  # Same that decoderTarget plus the <go>
        self.decoder_targets = None
        self.decoder_weights = None  # Adjust the learning to the target sentence size
        # Main operators
        self.loss = None
        self.optimize = None
        self.outputs = None  # Outputs of the network, list of probability for each words
        # new
        self.vocabulary_size = vocabulary_size
        self.start_token = start_token
        # Construct the graphs
        self.build_network()

    def create_rnn_cell(self):

        enc_dec_cell = tf.nn.rnn_cell.BasicLSTMCell(self.conf.hidden_size)
        # if not self.conf.test:
        #     enc_dec_cell = tf.layers.BatchNormalization(enc_dec_cell)
        return enc_dec_cell

    def build_network(self):
        # placeholder
        with tf.name_scope('placeholder_encoder'):
            self.encoder_inputs = [tf.placeholder(tf.int32, [None]) for _ in
                                   range(self.conf.max_length_encode)]
        with tf.name_scope('placeholder_decoder'):
            self.decoder_inputs = [tf.placeholder(tf.int32, [None], name='inputs') for _ in
                                   range(self.conf.max_length_decode)]
        self.decoder_targets = [tf.placeholder(tf.int32, [None], name='targets') for _ in
                                range(self.conf.max_length_decode)]
        self.decoder_weights = [tf.placeholder(tf.float32, [None], name='weights') for _ in
                                range(self.conf.max_length_decode)]

        # Creation of the rnn cell
        enc_dec_cell = tf.contrib.rnn.MultiRNNCell(
            [self.create_rnn_cell() for _ in range(self.conf.num_layers)])

        # embedding wrap
        decoder_outputs, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
            self.encoder_inputs,
            self.decoder_inputs,
            enc_dec_cell,
            self.vocabulary_size,
            self.vocabulary_size,
            embedding_size=self.conf.embedding_size,
            feed_previous=self.conf.test
        )

        # For testing only
        if self.conf.test:
            self.outputs = decoder_outputs
        # For training only
        else:
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss(
                decoder_outputs,
                self.decoder_targets,
                self.decoder_weights,
                self.vocabulary_size
            )
        tf.summary.scalar('loss', self.loss)  # Keep track of the cost
        # Initialize the optimizer
        self.optimize = tf.train.AdamOptimizer(learning_rate=self.conf.learning_rate).minimize(self.loss)

    def step(self, batch):
        feed_dict = {}
        ops = None
        if not self.conf.test:  # Training
            for i in range(self.conf.max_length_encode):  # input length
                feed_dict[self.encoder_inputs[i]] = batch.encoder_seqs[i]  # batch feed dict
            for i in range(self.conf.max_lenght_decode):
                feed_dict[self.decoder_inputs[i]] = batch.decoder_seqs[i]
                feed_dict[self.decoder_targets[i]] = batch.target_seqs[i]
                feed_dict[self.decoder_weights[i]] = batch.weights[i]

            ops = self.optimize, self.loss
        else:  # Testing (batchSize == 1)
            for i in range(self.conf.max_length_encode):
                feed_dict[self.encoder_inputs[i]] = batch.encoder_seqs[i]
            feed_dict[self.decoder_inputs[0]] = [self.start_token]
            ops = (self.outputs,)
        # Return one pass operator

        return ops, feed_dict
