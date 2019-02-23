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

def scatter_add_tensor(ref, indices, updates, name=None):
    """
    Adds sparse updates to a variable reference.

    This operation outputs ref after the update is done. This makes it easier to chain operations that need to use the
    reset value.

    Duplicate indices are handled correctly: if multiple indices reference the same location, their contributions add.

    Requires updates.shape = indices.shape + ref.shape[1:].
    :param ref: A Tensor. Must be one of the following types: float32, float64, int64, int32, uint8, uint16,
        int16, int8, complex64, complex128, qint8, quint8, qint32, half.
    :param indices: A Tensor. Must be one of the following types: int32, int64. A tensor of indices into the first
        dimension of ref.
    :param updates: A Tensor. Must have the same dtype as ref. A tensor of updated values to add to ref
    :param name: A name for the operation (optional).
    :return: Same as ref. Returned as a convenience for operations that want to use the updated values after the update
        is done.
    """
    with tf.name_scope(name, 'scatter_add_tensor', [ref, indices, updates]) as scope:
        # ref = tf.convert_to_tensor(ref, name='ref')
        # indices = tf.convert_to_tensor(indices, name='indices')
        # updates = tf.convert_to_tensor(updates, name='updates')
        ref_shape = tf.shape(ref, out_type=indices.dtype, name='ref_shape')
        scattered_updates = tf.scatter_nd(indices, updates, ref_shape, name='scattered_updates')
        print scattered_updates

        with tf.control_dependencies([tf.assert_equal(ref_shape, tf.shape(scattered_updates, out_type=indices.dtype))]):
            output = tf.add(ref, scattered_updates, name=scope)
        return output

O = 4
D = 5
T = 3

# batch_size = 2

a = [
	 [[1, 1, 1, 1, 1],
	  [2, 2, 2, 2, 2],
	  [10, 10, 10, 10, 10]],
	 [[1, 1, 1, 1, 1],
	  [2, 2, 2, 2, 2],
	  [10, 10, 10, 10, 10]]]

# obj_vecs = tf.reshape(tf.range(40), [2, O, D])

obj_vecs = tf.placeholder(tf.int32, [None, O, D])
pred_vecs = tf.zeros((T, D))

dtype = obj_vecs.dtype
print obj_vecs.shape[0]

edges = tf.placeholder(tf.int32, [None, T, 2])
batch_size = tf.shape(edges)[0]


s_idx = edges[:, :, 0]
o_idx = edges[:, :, 1]
print s_idx # (B, T)

cur_s_vecs = tf.batch_gather(obj_vecs, s_idx) # (B, T, D)
cur_o_vecs = tf.batch_gather(obj_vecs, o_idx)

# # raw_input()

# pooled_obj_vecs = tf.Variable(tf.zeros([batch_size*O, D], dtype=tf.int32), validate_shape=False)
pooled_obj_vecs = tf.zeros([batch_size*O, D], dtype=tf.int32)
# raw_input()

s_idx = s_idx + tf.reshape(tf.range(batch_size*O, delta=O, name='range'), [batch_size, 1])
o_idx = o_idx + tf.reshape(tf.range(batch_size*O, delta=O, name='range'), [batch_size, 1])


pooled_obj_vecs = tf.reshape(pooled_obj_vecs, [-1, D])
s_idx = tf.reshape(s_idx, [-1, 1])
o_idx = tf.reshape(o_idx, [-1, 1])
cur_s_vecs = tf.reshape(cur_s_vecs, [-1, D]) # (6, 5)
cur_o_vecs = tf.reshape(cur_o_vecs, [-1, D]) # (6, 5)

# print s_idx
# print cur_s_vecs
# print pooled_obj_vecs
# raw_input()

ref_shape = tf.shape(pooled_obj_vecs) # (8, 5)
s_scattered_updates = tf.scatter_nd(s_idx, cur_s_vecs, ref_shape)
o_scattered_updates = tf.scatter_nd(o_idx, cur_o_vecs, ref_shape)
        
pooled_obj_vecs = s_scattered_updates + o_scattered_updates

# pooled_obj_vecs = scatter_add_tensor(pooled_obj_vecs, s_idx, cur_s_vecs)
# pooled_obj_vecs = tf.scatter_add(pooled_obj_vecs, s_idx, cur_s_vecs)
# pooled_obj_vecs = tf.scatter_add(pooled_obj_vecs, o_idx, cur_o_vecs)
# print scattered_updates
# raw_input()

# obj_counts = tf.Variable(tf.zeros((batch_size*O, )))
# ones = tf.ones(batch_size*T)
# obj_counts = tf.scatter_add(obj_counts, s_idx, ones)
# obj_counts = tf.scatter_add(obj_counts, o_idx, ones)
# obj_counts = tf.clip_by_value(obj_counts, 1, 10000)

# obj_counts = tf.reshape(obj_counts, [-1, O])

# pooled_obj_vecs = tf.reshape(pooled_obj_vecs, [-1, D])
# a =  tf.cast(pooled_obj_vecs, tf.float32) / tf.reshape(obj_counts, (-1, 1))

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
_obj_vecs = np.arange(40).reshape([2, O, D])

# [1. 3. 1. 1.]
#  [2. 1. 2. 1.]]

feed_dict = {
	edges: _edges,
	obj_vecs: _obj_vecs,
}

# _s_idx = sess.run(s_idx, feed_dict)
_s_scattered_updates, _o_scattered_updates, _obj_vecs, _cur_s_vecs, _pooled_obj_vecs = sess.run([s_scattered_updates, o_scattered_updates, obj_vecs, cur_s_vecs, pooled_obj_vecs], feed_dict)
# _a, _obj_counts, _s_idx, _obj_vecs, _cur_s_vecs, _pooled_obj_vecs = sess.run([a, obj_counts, s_idx, obj_vecs, cur_s_vecs, pooled_obj_vecs], feed_dict)
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
# print _obj_counts
# print _s_idx
# print _a

print '-----_scattered_updates-----'
print _s_scattered_updates
print _o_scattered_updates
































