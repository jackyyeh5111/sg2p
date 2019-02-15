import tensorflow as tf
import numpy as np
x = tf.placeholder(tf.float32, shape=[None, 4])

# Use tf.shape() to get the runtime size of `x` in the 0th dimension.
zeros_dims = tf.stack([tf.shape(x)[0]*2, 7])

y = tf.fill(zeros_dims, 0.0)

y = tf.zeros(zeros_dims)

sess = tf.Session()
y_result = sess.run(y, feed_dict={x: np.random.rand(4, 4)})
print y_result.shape
print y_result
# ==> (4, 7)