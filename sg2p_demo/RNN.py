import tensorflow as tf
import numpy as np

class BasicLSTMCell(object):

  def __init__(self, embed_dim, hidden_dim, f_bias=1.0):
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.weight_initializer = tf.random_normal_initializer(stddev=0.1, seed=0)
        # self.weight_initializer = tf.orthogonal_initializer()
        self.f_const_initializer = tf.constant_initializer(f_bias)      
        self.const_initializer = tf.constant_initializer(0.0)      
        self.f_bias = f_bias 


  def create_recurrent_unit(self):
        # Weights and Bias for input and hidden tensor

      self.Wi = tf.get_variable('w_i', [self.embed_dim, self.hidden_dim], initializer=self.weight_initializer)
      self.Ui = tf.get_variable('u_i', [self.hidden_dim, self.hidden_dim], initializer=self.weight_initializer)
      self.bi = tf.get_variable('b_i', [self.hidden_dim], initializer=self.const_initializer)

      self.Wf = tf.get_variable('w_f', [self.embed_dim, self.hidden_dim], initializer=self.weight_initializer)
      self.Uf = tf.get_variable('u_f', [self.hidden_dim, self.hidden_dim], initializer=self.weight_initializer)
      self.bf = tf.get_variable('b_f', [self.hidden_dim], initializer=self.f_const_initializer)

      self.Wog = tf.get_variable('w_og', [self.embed_dim, self.hidden_dim], initializer=self.weight_initializer)
      self.Uog = tf.get_variable('u_og', [self.hidden_dim, self.hidden_dim], initializer=self.weight_initializer)
      self.bog = tf.get_variable('b_og', [self.hidden_dim], initializer=self.const_initializer)

      self.Wc = tf.get_variable('w_c', [self.embed_dim, self.hidden_dim], initializer=self.weight_initializer)
      self.Uc = tf.get_variable('u_c', [self.hidden_dim, self.hidden_dim], initializer=self.weight_initializer)
      self.bc = tf.get_variable('b_c', [self.hidden_dim], initializer=self.const_initializer)

      self.Wo = tf.get_variable('w_o', [self.hidden_dim, self.hidden_dim], initializer=self.weight_initializer)
      self.bo = tf.get_variable('b_o', [self.hidden_dim], initializer=self.const_initializer)


      def unit(inputs, state, is_pad=None):

          previous_hidden_state, c_prev = tf.unstack(state)
          # dropout_hidden_state = tf.nn.dropout(previous_hidden_state, 0.5)

          # Input Gate
          i = tf.sigmoid(
              tf.matmul(inputs, self.Wi) +
              tf.matmul(previous_hidden_state, self.Ui) + self.bi
          )

          # Forget Gate
          f = tf.sigmoid(
              tf.matmul(inputs, self.Wf) +
              tf.matmul(previous_hidden_state, self.Uf) + self.bf
          )

          # Output Gate
          o = tf.sigmoid(
              tf.matmul(inputs, self.Wog) +
              tf.matmul(previous_hidden_state, self.Uog) + self.bog
          )

          # New Memory Cell
          c_ = tf.nn.tanh(
              tf.matmul(inputs, self.Wc) +
              tf.matmul(previous_hidden_state, self.Uc) + self.bc
          )      

          if is_pad == None:  # in encoder of seq2seq, every sents pads to fixed size
              
              # Final Memory cell
              c = f * c_prev + i * c_ 
              # Current Hidden state
              current_hidden_state = o * tf.nn.tanh(c)

          else:
              c = tf.where(is_pad, c_prev, f * c_prev + i * c_)
              current_hidden_state = tf.where(is_pad, previous_hidden_state, o * tf.nn.tanh(c))

          output = tf.matmul(current_hidden_state, self.Wo) + self.bo

          # return output, tf.stack([current_hidden_state, c])
          return output, (current_hidden_state, c) 

      return unit
