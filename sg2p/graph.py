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

class WeightInit():

    def __init__(self):
        self.random_uniform = tf.random_uniform_initializer(-0.1, 0.1)
        self.xavier = tf.contrib.layers.xavier_initializer()

class MLP():

  def __init__(self, dim_list, activation='relu', batch_norm='none',
              dropout_ratio=0, final_nonlinearity=True):

    self.dropout_ratio = dropout_ratio

    self.layers = []
    for i in range(len(dim_list) - 1):
      dim_in, dim_out = dim_list[i], dim_list[i + 1]
      self.layers.append( Dense(dim_out, kernel_initializer=self.w_init.xavier) )

  def __call__(self, vec):

    for i, layer in enumerate(self.layers):
      vec = layer(vec)

      final_layer = (i == len(self.layers) - 2)

      vec = tf.nn.relu(vec)

      if self.dropout_ratio > 0:
        vec = tf.nn.dropout(vec, keep_prob=self.dropout_ratio)



class GraphTripleConv():
  """
  A single layer of scene graph convolution.
  """
  def __init__(self, input_dim, output_dim=None, hidden_dim=512,
               attr_hidden_dim=None, 
               pooling='avg', mlp_normalization=None, dropout_ratio=0.5, 
               use_gcv_mlayer=False, max_n_attr=3, use_attrs=False):
    
    if output_dim is None:
      output_dim = input_dim
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim
    self.max_n_attr = max_n_attr
    self.use_attrs = use_attrs

    if attr_hidden_dim is None:
      attr_hidden_dim = hidden_dim
    self.attr_hidden_dim = attr_hidden_dim


    self.w_init = WeightInit()
    
    assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
    self.pooling = pooling
    
    self.layer1 = Dense(hidden_dim, kernel_initializer=self.w_init.xavier)
    self.layer2 = Dense(2 * hidden_dim + output_dim, kernel_initializer=self.w_init.xavier)

    self.attr_layer1 = Dense(self.attr_hidden_dim, kernel_initializer=self.w_init.xavier)
    self.attr_layer2 = Dense(self.attr_hidden_dim, kernel_initializer=self.w_init.xavier)

    self.layer3 = Dense(hidden_dim, kernel_initializer=self.w_init.xavier)
    self.layer4 = Dense(output_dim, kernel_initializer=self.w_init.xavier)

    self.activation = tf.nn.relu

    self.dropout_ratio = dropout_ratio
    self.mlp_normalization = mlp_normalization
    self.use_gcv_mlayer = use_gcv_mlayer

  def _batch_norm(self, x, mode, name=None, reuse=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope=name, 
                                            reuse=reuse)


  def middle_layer(self, mode, vecs):
    if self.mlp_normalization:
      vecs = self._batch_norm(vecs, mode=mode)

    if mode == 'train':
      vecs = tf.nn.dropout(vecs, self.dropout_ratio)
    
    return vecs

  def __call__(self, mode, obj_vecs, pred_vecs, edges, attr_vecs=None, attrs_mask=None):
    """
    Inputs:
    - obj_vecs: FloatTensor of shape (O, D) giving vectors for all objects
    - pred_vecs: FloatTensor of shape (T, D) giving vectors for all predicates
    - edges: LongTensor of shape (T, 2) where edges[k] = [i, j] indicates the
      presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]
    - attr_vecs: FloatTensor of shape (O, 3, D) giving vectors for all objects  
    
    Outputs:
    - new_obj_vecs: FloatTensor of shape (O, D) giving new vectors for objects
    - new_pred_vecs: FloatTensor of shape (T, D) giving new vectors for predicates
    """
    assert mode in ['train', 'test']

    dtype = obj_vecs.dtype
    O, T = tf.shape(obj_vecs)[1], tf.shape(pred_vecs)[1]
    batch_size = tf.shape(obj_vecs)[0]

    Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim
    
    # Break apart indices for subjects and objects; these have shape (T,)
    s_idx = edges[:, :, 0]
    o_idx = edges[:, :, 1]

    # print s_idx
    # print o_idx
    # print obj_vecs

    # Get current vectors for subjects and objects; these have shape (T, Din)
    cur_s_vecs = tf.batch_gather(obj_vecs, s_idx)
    cur_o_vecs = tf.batch_gather(obj_vecs, o_idx)
    
    # print cur_s_vecs
    # print cur_o_vecs
    # print pred_vecs
    # raw_input()

    # Get current vectors for triples; shape is (T, 3 * Din)
    # Pass through net1 to get new triple vecs; shape is (T, 2 * H + Dout)
    cur_t_vecs = tf.concat([cur_s_vecs, pred_vecs, cur_o_vecs], axis=2)
    cur_t_vecs = self.activation( self.layer1(cur_t_vecs) )
    if self.use_gcv_mlayer:
      cur_t_vecs = self.middle_layer(mode, cur_t_vecs)
    
    new_t_vecs = self.activation( self.layer2(cur_t_vecs) )
    if self.use_gcv_mlayer:
      new_t_vecs = self.middle_layer(mode, new_t_vecs)


    # Break apart into new s, p, and o vecs; s and o vecs have shape (T, H) and
    # p vecs have shape (T, Dout)
    new_s_vecs = new_t_vecs[:, :, :H]
    new_p_vecs = new_t_vecs[:, :, H:(H+Dout)]
    new_o_vecs = new_t_vecs[:, :, (H+Dout):(2 * H + Dout)]
 
    print new_o_vecs
    

    # Allocate space for pooled object vectors of shape (O, H)
    pooled_obj_vecs = tf.zeros((batch_size, O, H), dtype=dtype)

    # trick for scatter_add (tf scatter_add cannot use batch_dim)
    s_idx = s_idx + tf.reshape(tf.range(batch_size*O, delta=O), [-1, 1])
    o_idx = o_idx + tf.reshape(tf.range(batch_size*O, delta=O), [-1, 1])
    
    pooled_obj_vecs = tf.reshape(pooled_obj_vecs, [-1, H])
    s_idx = tf.reshape(s_idx, [-1, 1])
    o_idx = tf.reshape(o_idx, [-1, 1])
    new_s_vecs = tf.reshape(new_s_vecs, [-1, H])
    new_o_vecs = tf.reshape(new_o_vecs, [-1, H])

    
    ref_shape = tf.shape(pooled_obj_vecs) # (8, 5)
    s_scattered = tf.scatter_nd(s_idx, new_s_vecs, ref_shape, name='s_scattered')
    o_scattered = tf.scatter_nd(o_idx, new_o_vecs, ref_shape, name='o_scattered')

    pooled_obj_vecs = s_scattered + o_scattered

    # Use scatter_add to sum vectors for objects that appear in multiple triples;
    # we first need to expand the indices to have shape (T, D)
    # pooled_obj_vecs = tf.scatter_add(pooled_obj_vecs, s_idx, new_s_vecs)
    # pooled_obj_vecs = tf.scatter_add(pooled_obj_vecs, o_idx, new_o_vecs)

    print 'pooled_obj_vecs'
    print pooled_obj_vecs


    if self.pooling == 'avg':
      # Figure out how many times each object has appeared, again using
      # some scatter_add trickery.
      obj_counts = tf.zeros((batch_size*O, ), dtype=dtype)
      ones = tf.ones(batch_size*T, dtype=dtype)
      ref_shape = tf.shape(obj_counts) # (8, 5)
      s_scattered = tf.scatter_nd(s_idx, ones, ref_shape, name='s_scattered')
      o_scattered = tf.scatter_nd(o_idx, ones, ref_shape, name='o_scattered')

      obj_counts = s_scattered + o_scattered


      # obj_counts = tf.Variable(tf.zeros((batch_size*O,), dtype=dtype), validate_shape=False)
      # ones = tf.ones(batch_size*T, dtype=dtype)
      # obj_counts = tf.scatter_add(obj_counts, s_idx, ones)
      # obj_counts = tf.scatter_add(obj_counts, o_idx, ones)

      # Divide the new object vectors by the number of times they
      # appeared, but first clamp at 1 to avoid dividing by zero;
      # objects that appear in no triples will have output vector 0
      # so this will not affect them.
      obj_counts = tf.clip_by_value(obj_counts, clip_value_min=1, clip_value_max=100000)
      print 'obj_counts'
      print obj_counts
      

      pooled_obj_vecs =  tf.cast(pooled_obj_vecs, tf.float32) / tf.reshape(obj_counts, (-1, 1))

    
    pooled_obj_vecs = tf.reshape(pooled_obj_vecs, [batch_size, O, H])

    print pooled_obj_vecs # (128, 30, 512)
    print new_p_vecs
    # raw_input()


    # attr_vecs: FloatTensor of shape (B, O, 3, D) giving vectors for all objects  
    # obj_vecs: FloatTensor of shape (B, O, D) giving vectors for all objects
    
    if self.use_attrs:
      tile_obj_vecs = tf.tile(obj_vecs, [1, 3, 1]) # (B, O*A, D)
      tile_obj_vecs = tf.reshape(tile_obj_vecs, [batch_size, O, self.max_n_attr, self.input_dim])

      cur_attr_vecs = tf.concat([tile_obj_vecs, attr_vecs], axis=3)
    
      cur_attr_vecs = self.attr_layer1(cur_attr_vecs)
      if self.use_gcv_mlayer:
        cur_attr_vecs = self.middle_layer(mode, cur_attr_vecs)
      
      cur_attr_vecs = self.attr_layer2(cur_attr_vecs)
      cur_attr_vecs = cur_attr_vecs * tf.expand_dims(attrs_mask, axis=3)

      pooled_attr_vecs = tf.reduce_mean(cur_attr_vecs, axis=2)

      # print pooled_attr_vecs
      # print pooled_obj_vecs
      # raw_input()

      # fuse attrs, objs (include relation info)
      pooled_fuse_vecs = tf.concat([pooled_attr_vecs, pooled_obj_vecs], axis=2)

    else: # use obj vecs only
      pooled_fuse_vecs = pooled_obj_vecs


    # Send pooled object vectors through net2 to get output object vectors,
    # of shape (O, Dout)
    pooled_fuse_vecs = self.activation( self.layer3(pooled_fuse_vecs) )
    if self.use_gcv_mlayer:
      pooled_fuse_vecs = self.middle_layer(mode, pooled_fuse_vecs)
    
    new_obj_vecs = self.activation( self.layer4(pooled_fuse_vecs) )

    print pooled_fuse_vecs # (128, 31, 512)
    print new_obj_vecs # (128, 31, H)
    # raw_input()


    return new_obj_vecs, new_p_vecs


class GraphTripleConvNet():
  """ A sequence of scene graph convolution layers  """
  def __init__(self, input_dim, num_layers=5, hidden_dim=512, pooling='avg',
               mlp_normalization='none'):

    self.num_layers = num_layers
    self.gconvs = nn.ModuleList()
    gconv_kwargs = {
      'input_dim': input_dim,
      'hidden_dim': hidden_dim,
      'pooling': pooling,
      'mlp_normalization': mlp_normalization,
    }
    for _ in range(self.num_layers):
      self.gconvs.append(GraphTripleConv(**gconv_kwargs))

  def forward(self, obj_vecs, pred_vecs, edges):
    for i in range(self.num_layers):
      gconv = self.gconvs[i]
      obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)
    return obj_vecs, pred_vecs