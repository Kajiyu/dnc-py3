import numpy as np
import tensorflow as tf
from dnc.controller import BaseController

"""
A 1-layer LSTM recurrent neural network with 256 hidden units
Note: the state of the LSTM is not saved in a variable becuase we want
the state to reset to zero on every input sequnece
"""

class RecurrentController(BaseController):

    def lstm_cell(self):
        return tf.contrib.rnn.BasicLSTMCell(256)

    def network_vars(self):
        self.stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(256) for _ in range(2)]
        )
        self.state = self.stacked_lstm.zero_state(self.batch_size, tf.float32)

    def network_op(self, X, state):
        X = tf.convert_to_tensor(X)
        return self.stacked_lstm(X, state)

    def get_state(self):
        return self.state

    def update_state(self, new_state):
        return tf.no_op()