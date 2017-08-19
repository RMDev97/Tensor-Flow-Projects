import numpy as np
import tensorflow as tf


def model_fn(features, labels, mode):
    # we will build a custom linear model with parameters to learn W and b
    W = tf.Variable([1.0], dtype=tf.float64)
    b = tf.Variable([1.0], dtype=tf.float64)
    model = W*features['x'] + b

    # define the square error cost function (where labels represents the dependent values)
    loss = tf.reduce_sum(tf.square(model - labels))

    # create the gradient descent training subgraph
    global_step = tf.train.get_global_step()
    alpha = 0.01
    optimiser = tf.train.GradientDescentOptimizer(alpha)
    # define the training graph to be composed of the gradient descent optimiser and a training step increment
    train = tf.group(optimiser.minimize(loss), tf.assign_add(global_step, 1))

    return tf.estimator.EstimatorSpec(mode=mode, predictions=model, loss=loss, train_op=train)

# retrieve the model for the estimator
estimator = tf.estimator.Estimator(model_fn=model_fn)

# define our data sets with the same training data as before
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

# define the input functions to the model which define which data sets are used for each evaluation of the model
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# train the model using the custom loss function and gradient descent algorithm
estimator.train(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r" % train_metrics)
print("eval metrics: %r" % eval_metrics)
