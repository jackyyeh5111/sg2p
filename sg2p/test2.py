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

O = 4
D = 5
T = 3

batch_size = 2

a = [
	 [[1, 1, 1, 1, 1],
	  [2, 2, 2, 2, 2],
	  [10, 10, 10, 10, 10]],
	 [[1, 1, 1, 1, 1],
	  [2, 2, 2, 2, 2],
	  [10, 10, 10, 10, 10]]]

obj_vecs = tf.reshape(tf.range(40), [2, O, D])
pred_vecs = tf.zeros((T, D))

dtype = obj_vecs.dtype
print obj_vecs.shape[0]

edges = tf.placeholder(tf.int32, [batch_size, T, 2])

s_idx = edges[:, :, 0]
o_idx = edges[:, :, 1]
print s_idx # (B, T)

cur_s_vecs = tf.batch_gather(obj_vecs, s_idx) # (B, T, D)
cur_o_vecs = tf.batch_gather(obj_vecs, o_idx)

# raw_input()
print np.array(a).shape 
pooled_obj_vecs = tf.Variable(tf.zeros([batch_size*O, D], dtype=tf.int32))
# raw_input()

s_idx = s_idx + tf.reshape(tf.range(batch_size*O, delta=O, name='range'), [batch_size, 1])
o_idx = o_idx + tf.reshape(tf.range(batch_size*O, delta=O, name='range'), [batch_size, 1])

# pooled_obj_vecs = tf.reshape(pooled_obj_vecs, [-1, D])
s_idx = tf.reshape(s_idx, [-1])
o_idx = tf.reshape(o_idx, [-1])
cur_s_vecs = tf.reshape(cur_s_vecs, [-1, D])


pooled_obj_vecs = tf.scatter_add(pooled_obj_vecs, s_idx, cur_s_vecs)
# pooled_obj_vecs = tf.scatter_add(pooled_obj_vecs, o_idx, cur_o_vecs)
print pooled_obj_vecs
# raw_input()

obj_counts = tf.Variable(tf.zeros((batch_size*O, )))
ones = tf.ones(batch_size*T)
obj_counts = tf.scatter_add(obj_counts, s_idx, ones)
obj_counts = tf.scatter_add(obj_counts, o_idx, ones)
obj_counts = tf.clip_by_value(obj_counts, 1, 10000)

obj_counts = tf.reshape(obj_counts, [-1, O])

pooled_obj_vecs = tf.reshape(pooled_obj_vecs, [-1, D])
a =  tf.cast(pooled_obj_vecs, tf.float32) / tf.reshape(obj_counts, (-1, 1))

############################################################
sess = tf.Session(config=tf_config)
init = tf.global_variables_initializer()
sess.run(init)

### run model ###
_edges = np.array([
				 [[1, 3],
				  [1, 2],
				  [2, 3]],

				 [[0, 1],
				  [0, 2],
				  [2, 3]]
				  ])


# [1. 3. 1. 1.]
#  [2. 1. 2. 1.]]

feed_dict = {
	edges: _edges,
}

# _obj_vecs, _cur_s_vecs = sess.run([obj_vecs, cur_s_vecs], feed_dict)
_a, _obj_counts, _s_idx, _obj_vecs, _cur_s_vecs, _pooled_obj_vecs = sess.run([a, obj_counts, s_idx, obj_vecs, cur_s_vecs, pooled_obj_vecs], feed_dict)
# _a, _obj_counts, _obj_vecs, _cur_s_vecs, _pooled_obj_vecs = sess.run([a, obj_counts, obj_vecs, cur_s_vecs, pooled_obj_vecs], feed_dict)


print '-----_obj_vecs-----'
print _obj_vecs 
print '-----_cur_s_vecs-----'
print _cur_s_vecs
print _cur_s_vecs.shape
# print _cur_s_vecs[:,0,:,:]
# print _cur_s_vecs[:,1,:,:]
print '-----_pooled_obj_vecs-----'
print _pooled_obj_vecs
print _obj_counts
print _s_idx
print _a
































