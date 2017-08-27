"""
This file includes a basic implementation of a Recurrent neural network using LSTM cells in order to facilitate
next word prediction given a sentence as the context.

The model itself is an unrolled implementation of the Penn-Tree-Bank model and works by learning a probability distribution
for each word and hence determining given a certain context how likely it is that a word will appear in the sentence next

The words themselves are embedded in some vector space as dense vectors such that the semantics of each words becomes an association
with other words by a distance metric
"""

import tensorflow as tf
import numpy as np


class PTBInputData(object):
    def __init__(self, batch_size, num_steps, epoch_size, input_data):
        """
        constructor for the data that represents the PTB dataset, the input_data param will be later mapped to
        a more suitable format for consumption by the RNN model
        :param batch_size:
        :param num_steps:
        :param epoch_size:
        :param input_data:
        """
        self.input_data = input_data
        self.epoch_size = epoch_size
        self.num_steps = num_steps
        self.batch_size = batch_size


class RecurrentNeuralNetwork(object):

    def __init__(self,
                 num_batches,
                 batch_size,
                 num_features,
                 input_data: PTBInputData,
                 hidden_size,
                 ltsm_keep_prob,
                 num_layers,
                 vocab_size):

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.ltsm_keep_prob = ltsm_keep_prob
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.num_features = num_features
        self.input_data = input_data
        self.hidden_size = hidden_size
        self.model = None

        # some helper functions for developing LTSM cells that use dropout for regularization
        def lstm():
            return tf.contrib.rnn.BasicLSTMCell(self.hidden_size,
                                                forget_bias=0.0,
                                                state_is_tuple=True,
                                                reuse=tf.get_variable_scope().reuse)

        def ltsm_with_dropout():
            return tf.contrib.rnn.DropoutWrapper(lstm(), output_keep_prob=self.ltsm_keep_prob)

        # define the Recurrent Neural Network based model
        rnn_cell = tf.contrib.rnn.MultiRNNCell(
            [ltsm_with_dropout() for _ in range(self.num_layers)],
            state_is_tuple=True)

        self.initial_state = rnn_cell.zero_state(self.batch_size, tf.float32)

        # define the input layer of the Network to convert the data into a vectorised embedding
        word_embedding = tf.get_variable("word_embedding", [self.vocab_size, self.hidden_size])
        self.inputs = tf.nn.dropout(tf.nn.embedding_lookup(word_embedding, self.input_data.input_data), self.ltsm_keep_prob)

        # finally construct the Recurrent Neural Network Model
        self.inputs = tf.unstack(self.inputs, num=self.input_data.num_steps, axis=1)
        outputs, state = tf.contrib.rnn.static_rnn(rnn_cell, initial_state=self.initial_state)
        self.outputs = outputs
        self.state = state

    def train(self):
        """
        The method to call when training the above model
        :return:
        """
        output = tf.reshape(tf.stack(axis=1, values=self.outputs), [-1, self.hidden_size])
        


