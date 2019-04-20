import os
import sys
import time
import cPickle as pickle
import numpy as np
import random
import json
import h5py
from RNN import BasicLSTMCell

import tensorflow as tf
from tensorflow.python.layers.core import Dense

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

a = np.array( [[1,0,0], [1,1,0], [1,1,1], [1,1,1]]).astype(np.int32)
b = np.array( [[10,0,0], [10,10,0], [10,10,10], [10,10,10]]).astype(np.int32)


a = tf.constant(a)
b = tf.constant(b)

c = tf.reduce_sum( tf.multiply( a, b ), 1, keep_dims=True )

# dot_a_b = tf.tensordot(a, b, 1)


# a = tf.constant(b)
# c = tf.reduce_sum(a, axis=1)

# bb = tf.constant(aa) / tf.reshape(c, (-1, 1))

sess = tf.Session(config=tf_config)
init = tf.global_variables_initializer()
sess.run(init)

print sess.run(a)
print sess.run(b)
print sess.run(c)
