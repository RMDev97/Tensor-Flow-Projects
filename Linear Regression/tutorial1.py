import tensorflow as tf

node1 = tf.placeholder(tf.float32)
node2 = tf.placeholder(tf.float32)

add_node = node1 + node2
double_add_node = add_node * 2

print(node1, node2)

session = tf.Session()
print(session.run(double_add_node, {node1: 3.0, node2: 4.0}))
print(session.run(double_add_node, {node1: [1.0, 2.0], node2: [5.0, 12.0]}))
