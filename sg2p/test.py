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

# class MLP:

#   def __init__(dim_list, activation='relu', batch_norm='none',
#               dropout=0, final_nonlinearity=True):

# 	  layers = []
# 	  for i in range(len(dim_list) - 1):
# 	    dim_in, dim_out = dim_list[i], dim_list[i + 1]
# 	    layers.append(Dense(dim_out))


# 	    # layers.append(nn.Linear(dim_in, dim_out))
# 	    final_layer = (i == len(dim_list) - 2)
	    
# 	    if not final_layer or final_nonlinearity:
# 	      if batch_norm == 'batch':
# 	        layers.append(nn.BatchNorm1d(dim_out))
# 	      if activation == 'relu':
# 	      	layers.append(tf.nn.relu)
# 	      # elif activation == 'leakyrelu':
# 	      #   layers.append(nn.LeakyReLU())
# 	    if dropout > 0:
# 	      layers.append(tf.nn.Dropout(p=dropout))

# 	  return nn.Sequential(*layers)

batch_size = 2

a = [[1, 1, 1, 1, 1],
	 [2, 2, 2, 2, 2],
	 [10, 10, 10, 10, 10]]

obj_vecs = tf.reshape(tf.range(20), [1, O, D])
pred_vecs = tf.zeros((T, D))
pooled_obj_vecs = tf.Variable(tf.zeros((O, D)))

dtype = obj_vecs.dtype
print obj_vecs.shape[0]

edges = tf.placeholder(tf.int32, [batch_size, None, 2])

s_idx = edges[:, :, 0]
o_idx = edges[:, :, 1]
print s_idx

cur_s_vecs = tf.gather(obj_vecs, s_idx, axis=1)
cur_o_vecs = tf.gather(obj_vecs, o_idx, axis=1)

print cur_s_vecs
raw_input()

# pooled_obj_vecs = tf.scatter_add(pooled_obj_vecs, s_idx, a)
# pooled_obj_vecs = tf.scatter_add(pooled_obj_vecs, o_idx, a)



# obj_counts = tf.Variable(tf.zeros((O, )))
# ones = tf.ones(T)
# obj_counts = tf.scatter_add(obj_counts, s_idx, ones)
# obj_counts = tf.scatter_add(obj_counts, o_idx, ones)
# obj_counts = tf.clip_by_value(obj_counts, 1, 10000)

# a =  pooled_obj_vecs / tf.reshape(obj_counts, (-1, 1))

############################################################
sess = tf.Session(config=tf_config)
init = tf.global_variables_initializer()
sess.run(init)

### run model ###
_edges = np.array([
				 [[0, 1],
				  [0, 2],
				  [1, 3]],
				 [[0, 1],
				  [0, 2],
				  [1, 3]]
				  ])

feed_dict = {
	edges: _edges,
}

# _a, _obj_counts, _obj_vecs, _cur_s_vecs, _pooled_obj_vecs = sess.run([a, obj_counts, obj_vecs, cur_s_vecs, pooled_obj_vecs], feed_dict)

# print _obj_vecs 
# print _cur_s_vecs
# print _pooled_obj_vecs
# print _obj_counts
# print _a
































