"""
This model will implement a basic linear classifier on the MNIST dataset using the ES model
1. We treat the inputs of the dataset as being a discrete path through the a 2 dimensional tensor space
2. We apply the lead-lag transformation in order to account for quadratic variation of the path and to
transform it into a continuous path
3. We calculate the signature of the path (truncated to some finite order N)
4. using the truncated Nth order signature of the lead-lag transformed path as a feature set,
we train a softmax classifier (trained with the ADAM optimisation algorithm)
5. we will then extract the details of this into a class which can be used elsewhere in code
"""

import tensorflow as tf
import numpy as np
import iisignature as sig

from ES.path_transforms import PathTransforms


class ESLinearClassifier:
    """
    For now we will assume that this classifier will be applied to 1 dimensional paths of the form (t,S_t)
    Where t is a monotonically increasing time parameter that acts as an index into the path
    (such that signatures determine uniqueness of paths up to tree like equivalence)
    """
    def __init__(self, input_training_data, input_training_labels, test_data, num_labels, alpha=0.1, order=2):
        self.lead_lag_input = []
        self.input = input_training_data
        self.labels = input_training_labels
        self.test_data = test_data
        self.input_signatures = []
        self.alpha = alpha
        self.order = order
        self.num_labels = num_labels
        self.model = None

    def train(self):
        # compute the lead-lag transform of the input path to account for 2-variation of the path
        for X in self.input:
            input_lead = PathTransforms.lead(X)
            input_lag = PathTransforms.lag(X)
            self.lead_lag_input.append(zip(input_lead, input_lag))

        # compute the signature for this path
        self.input_signatures = [list(sig.sig(np.array(stream), self.order)) for stream in self.lead_lag_input]

        # train a basic softmax classifier model on this new feature set
        num_features = (2 ** self.order) - 1
        x = tf.placeholder(tf.float32, [None, num_features])
        W = tf.Variable(tf.zeroes([num_features, self.num_labels]))
        b = tf.Variable(tf.zeroes([self.num_labels]))
        y = tf.placeholder(tf.float32, [self.num_labels])

        self.model = tf.nn.softmax(tf.matmul(x, W) + b)
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=tf.matmul(x, W) + b, labels=y))

        # perform the learning step with ADAM as the optimiser
        optimiser = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(cross_entropy_loss)

        num_iterations = 1000
        with tf.Session as session:
            for i in range(num_iterations):
                session.run(optimiser, {x: self.input_signatures, y: self.labels})

    def predict(self, input_path):
        # compute the signature of the input path (assumed to be a 1 dimensional path)
        lead_lag_path = zip(PathTransforms.lead(input_path), PathTransforms.lag(input_path))
        signature = sig.sig(lead_lag_path, self.order)
        num_features = (2 ** self.order) - 1
        x = tf.placeholder(tf.float32, [None, num_features])

        # use the model to predict a value for the label
        prediction = None
        with tf.Session as session:
            prediction = session.run(self.model, feed_dict={x: signature})
        return prediction









