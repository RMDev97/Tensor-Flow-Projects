import tensorflow as tf

# Our model parameters
# W is the weight matrix, b the bias constant and x the input variable, y will be the output variable
W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

# a simple linear model
linear_model = W * x + b

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

print("Before training:")
print(session.run(linear_model, {x: [1.0, 2.0, 3.0, 4.0]}))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

print("Calculating the loss function on this input:")
print(session.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# we will now proceed to perform an optimisation algorithm on this linear model to detemine the true values of W and b
# which minimise the loss function

# alpha is the learning rate of the gradient descent learning algorithm
alpha = 0.01
optimiser = tf.train.GradientDescentOptimizer(alpha)
train = optimiser.minimize(loss)

# run the training algorithm for N=1000 iterations
num_of_iterations = 10000
# reset the values of the linear model to be equal to their initial values
session.run(init)

# training set for the linear model
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

print("Now training the model via Gradient descent:")
for i in range(num_of_iterations):
    # perform a single iteration of the gradient descent algorithm
    session.run(train, {x: x_train, y: y_train})
    loss_value = session.run(loss, {x: x_train, y: y_train})
    print("Result after iteration %s: %s" % (i, loss_value))

# evaluate the training accuracy of the model
curr_W, curr_b, curr_loss = session.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s, b: %s, Loss: %s" % (curr_W, curr_b, curr_loss))

# now consider a different testing set of data
x_eval = [2., 5., 8., 1.]
y_eval = [-1.01, -4.1, -7, 0.]

# run the loss function on this new set of data
loss_val = session.run(loss, {x: x_eval, y: y_eval})
print("Value of loss function when out of sample data is provided: %s", loss_val)


