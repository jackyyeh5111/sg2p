import os
import sys
import time
import cPickle as pickle
import numpy as np
import random
import json
import h5py
import tensorflow as tf
from tensorflow.python.layers.core import Dense

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

pad_idx = 0
D = 5

a = tf.constant([
	 [[0, 0, 0, 0, 0],
	  [2, 2, 2, 2, 2],
	  [10, 10, 10, 10, 10]],
	 [[0, 0, 0, 0, 0],
	  [2, 2, 2, 2, 2],
	  [10, 10, 10, 10, 10]]])

a_where = tf.not_equal(a, 0)

b = tf.placeholder(tf.int32, [None])
O = tf.shape(b)[0]
obj_vecs = tf.reshape(b, [-1, 1])

obj_vecs = tf.tile(obj_vecs, [3, 1])

attr_vecs = tf.reshape(a, [-1, D])
concat = tf.concat([obj_vecs, attr_vecs], axis=1)

concat = tf.reshape(concat, [O, 3, D+1])

# s_idx = tf.reshape(tf.range(batch_size*O, delta=O, name='range'), [batch_size, 1])
t = tf.reduce_sum(a, 2)
mask = tf.to_int32(tf.not_equal(t, pad_idx))
mask = tf.expand_dims(mask, 2)

pooled_attr_vecs = tf.reduce_mean(concat * mask, axis=1)


sess = tf.Session(config=tf_config)
init = tf.global_variables_initializer()
sess.run(init)

bb = [1, 2]
feed = {
	b: bb,
}

print sess.run(a) #(2, 3, 5)
print sess.run(b, feed) #(2, 1)
print sess.run(obj_vecs, feed) #(2, 1)
print sess.run(concat, feed) #(2, 1)
print sess.run(mask)
print sess.run(mask).shape
print sess.run(concat*mask, feed)
print '-----pooled_attr_vecs-----'
print sess.run(pooled_attr_vecs, feed)
print pooled_attr_vecs.shape
print '-----a_where-----'
print sess.run(a_where)


# a = np.array([
# 	 [[0, 0, 0, 0, 0],
# 	  [2, 2, 2, 2, 2],
# 	  [10, 10, 10, 10, 10]],
# 	 [[0, 0, 0, 0, 0],
# 	  [2, 2, 2, 2, 2],
# 	  [10, 10, 10, 10, 10]]])

# idx = np.where(a==2)
# print idx
# a[idx] = 3
# print a












