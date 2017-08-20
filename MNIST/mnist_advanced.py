import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
# the output labels with one hot enabled will be rank 1 tensors of 10 dimensions where the output corresponding to a
# digit will be set to one
mnist_data = input_data.read_data_sets("MNIST_data", one_hot=True)
session = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 28*28])
y = tf.placeholder(tf.float32, [None, 10])


def weight_variable(shape):
    """
    we return a randomly initialized weight tensor (whose elements are normal distributed)
    to avoid having zero gradients
    :param shape: the shape of the returned weight tensor
    :return: a weight tensor variable (whose elements are normal distributed with mean 0, stddev 0.1)
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    we introduce a positively initialised bias to avoid dead neurons in the CNN
    :param shape: the shape of the bias tensor
    :return: a bias tensor variable (initialised to 0.1)
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def convolutional_layer_2d(x, W):
    """
    creates a convolutional layer for a convolutional neural network
    :param x: the input tensor from out dataset
    :param W: the weight tensor which will be learned in the optimisation stage
    :return: a convolutional layer with stride of 1 and padding of zeroes to ensure that the output layer
    has the same shape as the input
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_layer_2x2(x):
    """
    creates a NN maxpool layer for aggregating inputs to the input with the largest response
    :param x: the input tensor from the previous layer operation
    :return: a max pool layer over 2x2 blocks
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


"""
Create the Convolutional Neural Network
"""

# implement the first convolutional layer
# the weight tensor will have a 5x5 patch size (size of the filter for the layer), 1 input channel
# and 32 output channels, the bias will of size 32
W_convolutional_layer_1 = weight_variable([5, 5, 1, 32])
b_convolutional_layer_1 = bias_variable([32])

# we resize the input image data to be a rank 4 tensor where the image is 28x28 and has a single color channel
x_image = tf.reshape(x, [-1, 28, 28, 1])

convolutional_1 = tf.nn.relu(convolutional_layer_2d(x_image, W_convolutional_layer_1) + b_convolutional_layer_1)
max_pool_1 = max_pool_layer_2x2(convolutional_1)
# this results in an image size of 14x14

# implement the 2nd convolutional neural network
W_convolutional_layer_2 = weight_variable([5, 5, 32, 64])
b_convolutional_layer_2 = bias_variable([64])
convolutional_2 = tf.nn.relu(convolutional_layer_2d(max_pool_1, W_convolutional_layer_2) + b_convolutional_layer_2)
max_pool_2 = max_pool_layer_2x2(convolutional_2)

# densely connected (fully connected) layer for processing the 7x7 image
# where we have 1024 neurons each processing the 7x7 image
W_fully_connected_layer_1 = weight_variable([7*7*64, 1024])
b_fully_connected_layer_1 = bias_variable([1024])
max_pool_2_flat = tf.reshape(max_pool_2, [-1, 7*7*64])
fully_connected_layer_1 = tf.nn.relu(tf.matmul(max_pool_2_flat, W_fully_connected_layer_1) + b_fully_connected_layer_1)

# in order to avoid overfitting of the model and allow it generalise on new datasets we employ dropout
# as the regularisation method. We could also apply l1 and l2 norms to encourage sparsity in the network
keep_probability = tf.placeholder(tf.float32)
fully_connected_layer_1_dropout = tf.nn.dropout(fully_connected_layer_1, keep_probability)

# readout layer - performs the final mapping of the CNN computation to rank 1 10 dimensional output
W_fully_connected_layer_2 = weight_variable([1024, 10])
b_fully_connected_layer_2 = bias_variable([10])
output_convolutional_layer = tf.matmul(fully_connected_layer_1_dropout, W_fully_connected_layer_2) \
                             + b_fully_connected_layer_2

"""
Train and evaluate the accuracy of the model
"""
# first define the cross entropy loss function to be used with the ADAM optimisation method
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_convolutional_layer))

# represents a single iteration of the optimisation algorithm
alpha = 1e-4
train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy)

# define the accuracy metric
correctness_metric = tf.equal(tf.argmax(output_convolutional_layer, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctness_metric, tf.float32))

num_of_iterations = 20000
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(num_of_iterations):
        batch_xs, batch_ys = mnist_data.train.next_batch(50)
        # implement logging for every 100 iterations
        if i % 100 == 0:
            train_accuracy = session.run(accuracy, {x: batch_xs, y: batch_ys, keep_probability: 1.0})
            print("Accuracy of model at iteration %s: %s" % (i, train_accuracy))
        session.run(train_step, {x: batch_xs, y: batch_ys, keep_probability: 0.5})

    # print the final accuracy of the resulting model
    final_accuracy = session.run(accuracy, {x: mnist_data.test.images, y: mnist_data.test.labels, keep_probability: 1.0})
    print("Final test accuracy: %s" % final_accuracy)






