import tensorflow as tf
import numpy as np

# define the feature columns which will represent the input variables of the model (which is the input x)
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# define the linear regression estimator which we are going to use to estimate the value of y
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# define the training data we will use to train the linear regression model
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

num_iterations = 1000

# define the input functions for the model training algorithm
input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=num_iterations, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_eval}, y_eval, batch_size=4, num_epochs=num_iterations, shuffle=False)

print("Now training the model...")
# invoke the training algorithm
estimator.train(input_fn=input_fn, steps=num_iterations)
print("finished training the model!")

# evaluate the estimator versus the training data and test data
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

print("Final training metric: %r" % train_metrics)
print("Final evaluation metric: %r" % eval_metrics)

