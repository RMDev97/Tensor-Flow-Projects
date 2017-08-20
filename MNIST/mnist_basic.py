# a basic implementation of the MNIST classifier through multinomial logistic regression
import tensorflow as tf

# import the dataset called MNIST for handwritten digit classification
from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

# create the input placeholder variable which represents our independent variable
# here x is a rank 2 tensor with arbitrary size 1st dimension and 784 as the 2nd dimension
x = tf.placeholder(tf.float32, [None, 784])

# the linear weight matrix W and bias vector b are the parameters to be learnt in our model
# here we are predicting y as softmax(W*x + b)
# W is a [784, 10] rank 2 tensor and b is a [10] rank 1 tensor
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# we define the model by the softmax (multinomial logistic regression model) equation as stated above
model = tf.nn.softmax(tf.matmul(x, W) + b)

# our loss function will be the cross entropy function of our predictions weighted by the actual values
y = tf.placeholder(tf.float32, [None, 10])

# here the sum is taken over the second dimension of y (of size 10) and the mean over the batch size (first dimension)
# cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(model), reduction_indices=[1]))
# the above is numerically unstable (will do a test later) so we will resort to use the tensorflow implementation
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.matmul(x, W) + b, labels=y))

# define the training graph of the model to use gradient descent
alpha = 0.4
train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy_loss)

# run the gradient descent algorithm for a pre-specified number of iterations
num_of_iterations = 1000
batch_size = 100
session = tf.InteractiveSession()

tf.global_variables_initializer().run()

# here we are applying stochastic gradient descent by only using a portion of the training data (chosen at random)
# on each iteration
for i in range(num_of_iterations):
    batch_xs, batch_ys = mnist_data.train.next_batch(batch_size)
    session.run(train_step, {x: batch_xs, y: batch_ys})
    loss_value = session.run(cross_entropy_loss, {x: batch_xs, y: batch_ys})
    print("Value of the loss function (cross entropy) on iteration %s: %s" % (i, loss_value))

# we evaluate the predictive correctness of the model by measuring the deviation from the true values
correctness_metric = tf.equal(tf.argmax(model, 1), tf.argmax(y,1))
# the above returns a rank 1 tensor of booleans which we will cast to numbers and calculate the mean
accuracy = tf.reduce_mean(tf.cast(correctness_metric, tf.float32))

# the accuracy metric is then calculated on the test data
print(session.run(accuracy, {x: mnist_data.test.images, y: mnist_data.test.labels}))









