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


b = np.array( [[1,0,0], [1,1,0], [1,1,1]] )
aa = np.ones((2, 2, 2))
aa[0][0] = 0
# aa = np.array( [[10,0,0], [10,10,0], [10,10,10], [0, 0, 0]] )

# cc = tf.placeholder(tf.float32, [None, max_n_objs, box_feats_dim])

cc = tf.Variable(aa.astype(np.float32))
layer = Dense(5, kernel_initializer=tf.contrib.layers.xavier_initializer())
new_t_vecs = layer(cc) 

# a = tf.constant(b)
# c = tf.reduce_sum(a, axis=1)

# bb = tf.constant(aa) / tf.reshape(c, (-1, 1))

sess = tf.Session(config=tf_config)
init = tf.global_variables_initializer()
sess.run(init)

# print sess.run(bb)
print sess.run(new_t_vecs)




