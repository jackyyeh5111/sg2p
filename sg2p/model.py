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
from graph import GraphTripleConv
from tensorflow.python.layers.core import Dense

class WeightInit():

    def __init__(self):
        self.random_uniform = tf.random_uniform_initializer(-0.1, 0.1)
        self.xavier = tf.contrib.layers.xavier_initializer()

class Attention():
    
    def __init__(self, num_boxes, attention_dim, reuse=False):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        
        self.w_init = WeightInit()   
        
        self.num_boxes = num_boxes
        self.attention_dim = attention_dim

        with tf.variable_scope('attention_layer', reuse=reuse):
            self.linear_feature = Dense(attention_dim, kernel_initializer=self.w_init.xavier)
            self.linear_hidden = Dense(attention_dim, kernel_initializer=self.w_init.xavier)
            self.full_att = Dense(1, activation=tf.nn.relu, kernel_initializer=self.w_init.xavier)
            self.softmax = tf.nn.softmax
           

    def __call__(self, img_features, decoder_hidden):
        """
        Forward propagation.
        :param img_features: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """

        features_proj = self.linear_feature(img_features)  # (batch_size, num_pixels, attention_dim)
        att = self.linear_hidden(decoder_hidden)  # (batch_size, attention_dim)
        
        h_att = tf.nn.relu( features_proj + tf.expand_dims(att, 1)) # (N, L, D)
        out_att = tf.reshape( self.full_att(tf.reshape(h_att, [-1, self.attention_dim])), [-1, self.num_boxes])   # (N, L)
        alpha = self.softmax(out_att) # (batch_size, num_pixels)

        context = tf.reduce_sum(img_features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)

        return context, alpha


class SentRNN():
    def __init__(self,
                 hidden_size,
                 lstm_layer,
                 wordRNN_lstm_dim,
                 pooling_dim,
                 feats_dim,
                 topic_dim,
                 num_boxes,
                 dropout_ratio=0.5,
                 ctx2out=True,
                 S_max=6,
                 reuse=False):
        
        self.hidden_size = hidden_size
        self.lstm_layer = lstm_layer
        self.topic_dim = topic_dim
        self.ctx2out = ctx2out
        self.feats_dim = feats_dim
        self.num_boxes = num_boxes
        self.S_max = S_max

        self.w_init = WeightInit()   

        self.dropout_ratio = dropout_ratio

        with tf.variable_scope('SentRNN', reuse=reuse):
            self.init_h = Dense(hidden_size, kernel_initializer=self.w_init.xavier)
            self.init_c = Dense(hidden_size, kernel_initializer=self.w_init.xavier)
        
            self.w_h = Dense(hidden_size, kernel_initializer=self.w_init.xavier)
            self.w_ctx2out = Dense(hidden_size, kernel_initializer=self.w_init.xavier)

            self.attention_layer = Attention(num_boxes, feats_dim)  # attention network

            self.sent_LSTM = tf.nn.rnn_cell.LSTMCell(self.hidden_size, state_is_tuple=True, initializer=tf.orthogonal_initializer())

            self.logistic = Dense(2, kernel_initializer=self.w_init.random_uniform)
        
            self.fc1 = Dense(topic_dim, activation=tf.nn.relu, kernel_initializer=self.w_init.random_uniform, name='fc1')
            self.fc2 = Dense(wordRNN_lstm_dim*2, activation=tf.nn.relu, kernel_initializer=self.w_init.random_uniform, name='fc2')
           
    
    def _init_hidden_state(self, img_features, reuse=False):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param img_features: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        with tf.variable_scope('initial_lstm', reuse=reuse):
            mean_img_features = tf.reduce_mean(img_features, 1)
            h = self.init_h(mean_img_features)  # (batch_size, decoder_dim)
            c = self.init_c(mean_img_features)
            return c, h


    def _decode_step(self, h, context, dropout, reuse=False):

        with tf.variable_scope('sent_decode_step', reuse=reuse):

            if dropout:
                h = tf.nn.dropout(h, self.dropout_ratio)
            
            h_logits = self.w_h(h)

            if self.ctx2out:
                context = self.w_ctx2out(context)
                h_logits += context
                
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, self.dropout_ratio)

        return h_logits

    def _batch_norm(self, x, mode, name=None, reuse=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope=name, 
                                            reuse=reuse)



    def __call__(self, img_feats, hidden_state=None, mode='train', reuse=None):

        assert mode in ['train', 'test']

        dropout = True if mode == 'train' else False

        img_feats = self._batch_norm(img_feats, mode=mode, name='feats_batch_norm', reuse=reuse)

        if hidden_state == None:
            c, h = self._init_hidden_state(img_feats)  # (batch_size, decoder_dim)
        else:
            c, h = hidden_state

        context, alpha = self.attention_layer(img_feats, h)
        
        _, (c, h) = self.sent_LSTM(inputs=context, state=[c, h])

        sent_output = self._decode_step(h, context, dropout)

        pred_stop = self.logistic(sent_output)
            
        _hidden = tf.nn.relu( self.fc1(sent_output) )

        topic_vec = tf.nn.relu( self.fc2( _hidden ) )

        return pred_stop, topic_vec, context, alpha, (h, c)



class Regions_Hierarchical():
    def __init__(self, word2idx,
                       batch_size,
                       pretrained_embed_matrix,
                       sentRNN_lstm_dim,
                       wordRNN_lstm_dim,
                       feats_dim, # =gconv_dim
                       use_box_feats,
                       n_objs,
                       max_n_objs,
                       max_n_rels,
                       use_attrs,
                       n_preds=50,
                       n_attrs=400,
                       embedding_dim=100,
                       box_feats_dim=2048,
                       # pred_embed_dim=32,
                       # obj_embed_dim=100,
                       # gconv_dim=128,
                       gconv_hidden_dim=512,
                       gconv_pooling='avg',
                       gconv_num_layers=5,

                       num_boxes=50,
                       project_dim=1024,
                       topic_dim=1024,
                       S_max=6,
                       N_max=30,
                       sentRNN_numlayer=1, 
                       wordRNN_numlayer=1,
                       encoder_lstm_dim=512, 
                       dropout=True,
                       ctx2out=True,
                       selector=False,
                       alpha_c=0.0):

            self.vocab_size = len(word2idx)
            self.pad_idx = word2idx["<pad>"]
            self.start_idx = word2idx["<bos>"]
            self.batch_size = batch_size
            self.num_boxes = num_boxes 
            self.feats_dim = feats_dim 
            self.project_dim = project_dim 
            self.S_max = S_max 
            self.N_max = N_max 
            self.embed_dim = pretrained_embed_matrix.shape[1]
            self.max_n_objs = max_n_objs
            self.max_n_rels = max_n_rels
            self.use_box_feats = use_box_feats
            self.use_attrs = use_attrs

            topic_dim = wordRNN_lstm_dim * 2

            self.w_init = WeightInit()

            self.H = wordRNN_lstm_dim
            self.L = num_boxes

            if self.use_box_feats:
                self.D = feats_dim + box_feats_dim
            else:
                self.D = feats_dim
            

            self.ctx2out = ctx2out
            self.alpha_c = alpha_c


            self.sentRNN = SentRNN(  sentRNN_lstm_dim,
                                     sentRNN_numlayer,
                                     wordRNN_lstm_dim,
                                     project_dim,
                                     self.D,
                                     topic_dim,
                                     num_boxes= max_n_objs)


            


            self.sentRNN_lstm_dim = sentRNN_lstm_dim 
            self.topic_dim = topic_dim 
            self.wordRNN_lstm_dim = wordRNN_lstm_dim 
            self.encoder_lstm_dim = encoder_lstm_dim

            self.sentRNN_numlayer = sentRNN_numlayer
            self.wordRNN_numlayer = wordRNN_numlayer

            self.selector = selector
            self.dropout = dropout

            self.attention_layer = Attention(num_boxes, feats_dim)  # attention network
            
            # logistic classifier
            self.logistic_Theta_W = tf.Variable(tf.random_uniform([sentRNN_lstm_dim, 2], -0.1, 0.1), name='logistic_Theta_W')
            self.logistic_Theta_b = tf.Variable(tf.zeros(2), name='logistic_Theta_b')

            # fc1_W: 512 x 1024, fc1_b: 1024
            # fc2_W: 1024 x 1024, fc2_b: 1024
            self.fc1_W = tf.Variable(tf.random_uniform([sentRNN_lstm_dim, topic_dim], -0.1, 0.1), name='fc1_W')
            self.fc1_b = tf.Variable(tf.zeros(topic_dim), name='fc1_b')
            self.fc2_W = tf.Variable(tf.random_uniform([topic_dim, wordRNN_lstm_dim*2], -0.1, 0.1), name='fc2_W')
            self.fc2_b = tf.Variable(tf.zeros(wordRNN_lstm_dim*2), name='fc2_b')

            
            # word LSTM
            # self.drop_prob = 0.0
            # word_cells = []
            # for _ in range(self.wordRNN_numlayer):
            #     word_cell = tf.contrib.rnn.BasicLSTMCell(self.wordRNN_lstm_dim, state_is_tuple=True)
            #     word_cell = tf.nn.rnn_cell.DropoutWrapper(word_cell, output_keep_prob=1-self.drop_prob)
            #     word_cells.append(word_cell)
                
            # self.word_LSTM = tf.contrib.rnn.MultiRNNCell(cells=word_cells, state_is_tuple=True)
            # self.word_LSTM = tf.contrib.rnn.BasicLSTMCell(wordRNN_lstm_dim, state_is_tuple=True)
            self.word_LSTM = tf.nn.rnn_cell.LSTMCell(self.wordRNN_lstm_dim, state_is_tuple=True, initializer=tf.orthogonal_initializer())

            self.embed_word_W = tf.Variable(tf.random_uniform([wordRNN_lstm_dim, self.vocab_size], -0.1,0.1), name='embed_word_W')
            self.embed_word_b = tf.Variable(tf.zeros([self.vocab_size]), name='embed_word_b')


            # placeholder
            # self.densecap_feats = tf.placeholder(tf.float32, [batch_size, self.num_boxes, self.feats_dim])
            self.box_feats = tf.placeholder(tf.float32, [None, max_n_objs, box_feats_dim])

            # receive the [continue:0, stop:1] lists
            # example: [0, 0, 0, 0, 1, 1], it means this paragraph has five sentences
            self.num_distribution = tf.placeholder(tf.int32, [batch_size, self.S_max])

            # receive the ground truth words, which has been changed to idx use word2idx function
            self.captions = tf.placeholder(tf.int32, [None, self.S_max, self.N_max+1])
            self.objs = tf.placeholder(tf.int32, [None, max_n_objs+1])
            self.triples = tf.placeholder(tf.int32, [None, max_n_rels, 3])
            self.attrs = tf.placeholder(tf.int32, [None, max_n_objs+1, 3]) # align with objs

            # self.caption_labels = tf.placeholder(tf.int32, [None, self.S_max, self.label_size])

            self.embed_initializer = tf.constant_initializer(pretrained_embed_matrix)

            self.Wemb = tf.get_variable("embedding",
                                        # 0,1,2 for pad sos eof respectively.
                                        [self.vocab_size, pretrained_embed_matrix.shape[1]],
                                        initializer=self.embed_initializer,
                                        trainable=False,
                                        dtype=tf.float32)


            self.obj_embeddings = tf.get_variable("obj_embeddings",
                                        # n_objs+1 for padding 
                                        [n_objs+1, embedding_dim],
                                        initializer=self.w_init.random_uniform)

            self.pred_embeddings = tf.get_variable("pred_embeddings",
                                        # 0,1,2 for pad sos eof respectively.
                                        [n_preds, embedding_dim],
                                        initializer=self.w_init.random_uniform)

            self.attr_embeddings = tf.get_variable("attr_embeddings",
                                        # n_objs+1 for padding 
                                        [n_attrs+1, embedding_dim],
                                        initializer=self.w_init.random_uniform)

            gconv_kwargs = {
                'input_dim': embedding_dim,
                'output_dim': feats_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'mlp_normalization': None,
                'use_attrs': use_attrs,
              }
            self.gconv = GraphTripleConv(**gconv_kwargs)

            self.weight_initializer = tf.contrib.layers.xavier_initializer()
            self.const_initializer = tf.constant_initializer(0.0)

    def _decode_lstm(self, h, context, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)

            h_logits = tf.matmul(h, w_h) + b_h # (batch_size, time_steps, hidden_cell_sizes)

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.H], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)
                
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)

            return h_logits
            
    def build_model(self, S_max, semi=False, reuse=False):

        # features = self.densecap_feats # (50, 4096)  
        objs = self.objs
        triples = self.triples  
        attrs = self.attrs 
        captions = self.captions
        box_feats = self.box_feats
        batch_size = tf.shape(objs)[0]

        captions_mask = tf.to_float(tf.not_equal(captions, self.pad_idx))
        captions_length = tf.reduce_sum(captions_mask, 2)
        sents_mask = tf.to_float(tf.not_equal(self.num_distribution, 0))

        # tf.split(triples, [2, 1], axis=2)
        edges, p =  triples[:, :, :2], triples[:, :, 2]  

        obj_vecs = tf.nn.embedding_lookup(self.obj_embeddings, objs)
        pred_vecs = tf.nn.embedding_lookup(self.pred_embeddings, p)

        if self.use_attrs:
            attr_vecs = tf.nn.embedding_lookup(self.attr_embeddings, attrs) 

            # build attrs_mask
            padding_attr = 400
            attrs_mask = tf.to_float(tf.not_equal(attrs, padding_attr))

            # graph convolution 
            obj_vecs, pred_vecs = self.gconv('train', obj_vecs, pred_vecs, edges, attr_vecs, attrs_mask)
        else:
            obj_vecs, pred_vecs = self.gconv('train', obj_vecs, pred_vecs, edges)

        obj_vecs = obj_vecs[:, :self.max_n_objs] # last idx is padding, ignore it!


        if self.use_box_feats:
            features = tf.concat([obj_vecs, box_feats], axis=2)
        else:
            features = obj_vecs


        print obj_vecs
        print tf.concat([obj_vecs, pred_vecs], axis=1)
        # raw_input()
        # features = obj_vecs
        print features
        
        # features = tf.concat([obj_vecs, pred_vecs], axis=1)

        # raw_input()
        # mask objs by 282
        # ----------------------------------------------------


        loss = 0.0
        loss_sent = tf.constant(0.0)
        loss_word = 0.0
        loss_label = tf.constant(0.0)
        lambda_sent = 5.0
        lambda_word = 1.0
        alpha_reg = tf.constant(0.0)

        alpha_list = []
        # reviewer_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.encoder_lstm_dim)
       
        print 'Start build model:'
        sent_hidden_state = None
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for i in range(0, S_max):
                print 'here %d' % i

                pred_stop, topic_vec, context, alpha, sent_hidden_state = self.sentRNN(features, sent_hidden_state, mode='train', reuse=(i!=0))

                # with tf.variable_scope('sent_LSTM', reuse=reuse or (i!=0)):
                #     _, (c, h) = self.sent_LSTM(inputs=context, state=[c, h])
                
                # sent_output = self._decode_lstm(h, context, dropout=self.dropout, reuse=reuse or (i!=0))

                # with tf.name_scope('fc1'):
                #     hidden1 = tf.nn.relu( tf.matmul(sent_output, self.fc1_W) + self.fc1_b )
                # with tf.name_scope('fc2'):
                #     sent_topic_vec = tf.nn.relu( tf.matmul(hidden1, self.fc2_W) + self.fc2_b )

                # sent loss
                
                # with tf.name_scope('sent_loss'):
                #     # sentRNN_logistic_mu = tf.nn.xw_plus_b( sent_output, self.logistic_Theta_W, self.logistic_Theta_b )
                #     sentRNN_label = tf.stack([ 1 - self.num_distribution[:, i], self.num_distribution[:, i] ])
                #     sentRNN_label = tf.transpose(sentRNN_label)
                #     sentRNN_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred_stop, labels=sentRNN_label)
                #     sentRNN_loss = tf.reduce_sum(sentRNN_loss)/self.batch_size
                #     loss += sentRNN_loss * lambda_sent
                #     loss_sent += sentRNN_loss
                    

                # wordRNN state
                topic = tf.contrib.rnn.LSTMStateTuple(topic_vec[:, 0:self.wordRNN_lstm_dim], topic_vec[:, self.wordRNN_lstm_dim:])
                # word_c, word_h = [topic] * self.wordRNN_numlayer
                word_c, word_h = topic


                for j in range(0, self.N_max):
                    if j > 0:
                        tf.get_variable_scope().reuse_variables()

               
                    current_embed = tf.nn.embedding_lookup(self.Wemb, captions[:, i, j])

                    # print "current_embed:", current_embed
                    
                    with tf.variable_scope('word_LSTM', reuse=(j!=0)):
                        _, (word_c, word_h) = self.word_LSTM(current_embed, state=[word_c, word_h])
                        word_output = self._decode_lstm(word_h, context, dropout=self.dropout, reuse=reuse or (j!=0))
                        logit_words = tf.nn.xw_plus_b(word_output, self.embed_word_W, self.embed_word_b)

                    labels = tf.reshape(captions[:, i, j+1], [-1, 1])
                    indices = tf.reshape(tf.range(0, self.batch_size, 1), [-1, 1])
                    concated = tf.concat([indices, labels], 1)
                    onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.vocab_size]), 1.0, 0.0)

                    
                    # At each timestep the hidden state of the last LSTM layer is used to predict a distribution
                    # over the words in the vocbulary
                    with tf.name_scope('word_loss'):
                        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
                        cross_entropy = cross_entropy * captions_mask[:, i, j]
                        loss_wordRNN = tf.reduce_sum(cross_entropy) / self.batch_size
                        loss += loss_wordRNN * lambda_word
                        loss_word += loss_wordRNN

        return loss, loss_sent, loss_word

    
    def build_sampler(self, reuse):

        # features = self.densecap_feats # (50, 4096)  
        objs = self.objs
        triples = self.triples  
        box_feats = self.box_feats
        attrs = self.attrs 
        captions = self.captions
        batch_size = tf.shape(objs)[0]

        captions_mask = tf.to_float(tf.not_equal(captions, self.pad_idx))
        captions_length = tf.reduce_sum(captions_mask, 2)
        sents_mask = tf.to_float(tf.not_equal(self.num_distribution, 0))

        # tf.split(triples, [2, 1], axis=2)
        edges, p =  triples[:, :, :2], triples[:, :, 2]  

        obj_vecs = tf.nn.embedding_lookup(self.obj_embeddings, objs)
        pred_vecs = tf.nn.embedding_lookup(self.pred_embeddings, p)

        if self.use_attrs:
            attr_vecs = tf.nn.embedding_lookup(self.attr_embeddings, attrs) 

            # build attrs_mask
            padding_attr = 400
            attrs_mask = tf.to_float(tf.not_equal(attrs, padding_attr))

            # graph convolution 
            obj_vecs, pred_vecs = self.gconv('test', obj_vecs, pred_vecs, edges, attr_vecs, attrs_mask)
        else:
            obj_vecs, pred_vecs = self.gconv('test', obj_vecs, pred_vecs, edges)

        obj_vecs = obj_vecs[:, :self.max_n_objs] # last idx is padding, ignore it!


        if self.use_box_feats:
            features = tf.concat([obj_vecs, box_feats], axis=2)
        else:
            features = obj_vecs

        print obj_vecs

        # features = obj_vecs
        # features = tf.concat([obj_vecs, pred_vecs], axis=1)
        

        # save the generated paragraph to list, here I named generated_sents
        generated_paragraph = []
        total_pred_labels = []
        pred_re = []
        alpha_list = []
        

        # Start build the generation model
        print ('Start build the generation model:')
        sent_hidden_state = None
        for i in range(0, self.S_max):

            if reuse == True:
                tf.get_variable_scope().reuse_variables()
            
            # context, alpha = self.attention_layer(features, h)
            # context, alpha = self._attention_layer(features, features_proj, h, reuse=reuse or (i!=0))
            # alpha_list.append(alpha)

            # if self.selector:
            #     context, beta = self._selector(context, h, reuse=reuse or (i!=0))

            pred_stop, topic_vec, context, alpha, sent_hidden_state = self.sentRNN(features, sent_hidden_state, mode='test', reuse=(i!=0))
            # with tf.variable_scope('sent_LSTM', reuse=reuse or (i!=0)):
            #     _, (c, h) = self.sent_LSTM(inputs=context, state=[c, h])

            # sent_output = self._decode_lstm(h, context, reuse=reuse or (i!=0))

            # with tf.name_scope('fc1'):
            #     hidden1 = tf.nn.relu( tf.matmul(sent_output, self.fc1_W) + self.fc1_b )
            # with tf.name_scope('fc2'):
            #     sent_topic_vec = tf.nn.relu( tf.matmul(hidden1, self.fc2_W) + self.fc2_b )

            
            # sentRNN_logistic_mu = tf.nn.xw_plus_b(sent_output, self.logistic_Theta_W, self.logistic_Theta_b)
            pred = tf.nn.softmax(pred_stop)
            pred_re.append(pred)

            # save the generated sentence to list, named generated_sent
            generated_sent = []

            # wordRNN state
            topic = tf.contrib.rnn.LSTMStateTuple(topic_vec[:, 0:self.wordRNN_lstm_dim], topic_vec[:, self.wordRNN_lstm_dim:])
            # word_c, word_h = [topic] * self.wordRNN_numlayer
            word_c, word_h = topic
                
            # word RNN, unrolled to N_max time steps
            for j in range(0, self.N_max):
                if j > 0:
                    tf.get_variable_scope().reuse_variables()

                if j == 0:
                    # get word embedding of BOS (index = 0)
                    current_embed = tf.nn.embedding_lookup(self.Wemb, tf.fill([tf.shape(self.objs)[0]], self.start_idx) )

                with tf.variable_scope('word_LSTM', reuse=reuse or (j!=0)):
                    _, (word_c, word_h) = self.word_LSTM(current_embed, state=[word_c, word_h])

                    word_output = self._decode_lstm(word_h, context, reuse=reuse or (j!=0))
                    logit_words = tf.nn.xw_plus_b(word_output, self.embed_word_W, self.embed_word_b)
                
                next_token = tf.argmax(logit_words, 1)
                generated_sent.append(next_token)
                current_embed = tf.nn.embedding_lookup(self.Wemb, next_token)


            generated_paragraph.append(generated_sent)

        generated_paragraph = tf.transpose(generated_paragraph, perm=[2, 0, 1]) # [batch_size, S_max, N_max]
        pred_re = tf.transpose(pred_re, perm=[1, 0, 2]) # [batch_size, S_max, N_max]
        
        return generated_paragraph, pred_re





