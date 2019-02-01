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

class Regions_Hierarchical():
    def __init__(self, word2idx,
                       batch_size,
                       pretrained_embed_matrix,
                       sentRNN_lstm_dim,
                       wordRNN_lstm_dim,
                       num_boxes=50,
                       feats_dim=4096,
                       project_dim=1024,
                       sentRNN_FC_dim=1024,
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

            self.H = sentRNN_lstm_dim
            self.L = num_boxes
            self.D = feats_dim
            self.ctx2out = ctx2out
            self.alpha_c = alpha_c


            self.sentRNN_lstm_dim = sentRNN_lstm_dim 
            self.sentRNN_FC_dim = sentRNN_FC_dim 
            self.wordRNN_lstm_dim = wordRNN_lstm_dim 
            self.encoder_lstm_dim = encoder_lstm_dim

            self.sentRNN_numlayer = sentRNN_numlayer
            self.wordRNN_numlayer = wordRNN_numlayer

            self.selector = selector
            self.dropout = dropout

            self.label_size = 7268
            # self.label_size = 7699

            # with tf.device('/cpu:0'):
            #     self.Wemb = tf.Variable(tf.random_uniform([self.vocab_size, word_embed_dim], -0.1, 0.1), name='Wemb')

            # regionPooling_W shape: 4096 x 1024
            # regionPooling_b shape: 1024
            self.regionPooling_W = tf.Variable(tf.random_uniform([feats_dim, project_dim], -0.1, 0.1), name='regionPooling_W')
            self.regionPooling_b = tf.Variable(tf.zeros([project_dim]), name='regionPooling_b')

            # sentence LSTM
            # self.sent_LSTM = tf.contrib.rnn.BasicLSTMCell(sentRNN_lstm_dim, state_is_tuple=True)
            self.sent_LSTM = tf.nn.rnn_cell.LSTMCell(self.sentRNN_lstm_dim, state_is_tuple=True, initializer=tf.orthogonal_initializer())

            # logistic classifier
            self.logistic_Theta_W = tf.Variable(tf.random_uniform([sentRNN_lstm_dim, 2], -0.1, 0.1), name='logistic_Theta_W')
            self.logistic_Theta_b = tf.Variable(tf.zeros(2), name='logistic_Theta_b')

            # fc1_W: 512 x 1024, fc1_b: 1024
            # fc2_W: 1024 x 1024, fc2_b: 1024
            self.fc1_W = tf.Variable(tf.random_uniform([sentRNN_lstm_dim, sentRNN_FC_dim], -0.1, 0.1), name='fc1_W')
            self.fc1_b = tf.Variable(tf.zeros(sentRNN_FC_dim), name='fc1_b')
            self.fc2_W = tf.Variable(tf.random_uniform([sentRNN_FC_dim, wordRNN_lstm_dim*2], -0.1, 0.1), name='fc2_W')
            self.fc2_b = tf.Variable(tf.zeros(wordRNN_lstm_dim*2), name='fc2_b')

            self.label_W = tf.Variable(tf.random_uniform([sentRNN_lstm_dim, self.label_size], -0.1, 0.1), name='label_W')
            self.label_b = tf.Variable(tf.zeros(self.label_size), name='label_b')

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
            self.densecap_feats = tf.placeholder(tf.float32, [None, self.num_boxes, self.feats_dim])

            # receive the [continue:0, stop:1] lists
            # example: [0, 0, 0, 0, 1, 1], it means this paragraph has five sentences
            self.num_distribution = tf.placeholder(tf.int32, [None, self.S_max])

            # receive the ground truth words, which has been changed to idx use word2idx function
            self.captions = tf.placeholder(tf.int32, [None, self.S_max, self.N_max+1])
            self.coco_captions = tf.placeholder(tf.int32, [None, 1, self.N_max+1])

            # self.caption_labels = tf.placeholder(tf.int32, [None, self.S_max, self.label_size])

            self.embed_initializer = tf.constant_initializer(pretrained_embed_matrix)

            self.Wemb = tf.get_variable("embedding",
                                        # 0,1,2 for pad sos eof respectively.
                                        [self.vocab_size, pretrained_embed_matrix.shape[1]],
                                        initializer=self.embed_initializer,
                                        trainable=False,
                                        dtype=tf.float32)



            self.weight_initializer = tf.contrib.layers.xavier_initializer()
            self.const_initializer = tf.constant_initializer(0.0)


    def _get_initial_lstm(self, features, reuse=None):
        with tf.variable_scope('initial_lstm', reuse=reuse):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)    # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])   # (N, L)
            alpha = tf.nn.softmax(out_att)
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
            return context, alpha

    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')    # (N, 1)
            context = tf.multiply(beta, context, name='selected_context')
            return context, beta

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.embed_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def _decode_lstm(self, h, context, dropout=False, reuse=False):
        # h.shape (batch_size, time_steps, hidden_cell_sizes)
        # context.shape (batch_size, 4096)
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            # w_out = tf.get_variable('w_out', [self.H, self.H], initializer=self.weight_initializer)
            # b_out = tf.get_variable('b_out', [self.H], initializer=self.const_initializer)

            # print h

            if dropout:
                h = tf.nn.dropout(h, 0.5)

            h_logits = tf.matmul(h, w_h) + b_h # (batch_size, time_steps, hidden_cell_sizes)

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.H], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)
                # h_logits += tf.expand_dims(tf.matmul(context, w_ctx2out), 1) 

            # if self.prev2out:
            #     h_logits += x
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)

            # out_logits = tf.matmul(h_logits, w_out) + b_out
            return h_logits

    def _batch_norm(self, x, mode='train', name=None, reuse=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope=(name+'batch_norm'), 
                                            reuse=reuse)



    def _project_features(self, features, reuse=None):
        with tf.variable_scope('project_features', reuse=reuse):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj

            
    def build_model(self, S_max, semi=False, reuse=False):

        features = self.densecap_feats # (50, 4096)

        if semi == True:
            captions = self.coco_captions
        else:
            captions = self.captions

        captions_mask = tf.to_float(tf.not_equal(captions, self.pad_idx))
        captions_length = tf.reduce_sum(captions_mask, 2)

        sents_mask = tf.to_float(tf.not_equal(self.num_distribution, 0))
        
        # captions_labels_masks = tf.to_float(tf.not_equal(self.num_distribution, self.pad_idx))

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='train', name='dense_features', reuse=reuse)

        c, h = self._get_initial_lstm(features=features, reuse=reuse)
        # x = self._word_embedding(inputs=captions_in)
        features_proj = self._project_features(features=features, reuse=reuse)

        
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
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for i in range(0, S_max):
                print 'here %d' % i
                context, alpha = self._attention_layer(features, features_proj, h, reuse=reuse or (i!=0))
                alpha_list.append(alpha)

                if self.selector:
                    context, beta = self._selector(context, h, reuse=reuse or (i!=0))


                with tf.variable_scope('sent_LSTM', reuse=reuse or (i!=0)):
                    _, (c, h) = self.sent_LSTM(inputs=context, state=[c, h])
                
                sent_output = self._decode_lstm(h, context, dropout=self.dropout, reuse=reuse or (i!=0))

                with tf.name_scope('fc1'):
                    hidden1 = tf.nn.relu( tf.matmul(sent_output, self.fc1_W) + self.fc1_b )
                with tf.name_scope('fc2'):
                    sent_topic_vec = tf.nn.relu( tf.matmul(hidden1, self.fc2_W) + self.fc2_b )

                # sent loss
                if semi == False:
                    with tf.name_scope('sent_loss'):
                        sentRNN_logistic_mu = tf.nn.xw_plus_b( sent_output, self.logistic_Theta_W, self.logistic_Theta_b )
                        sentRNN_label = tf.stack([ 1 - self.num_distribution[:, i], self.num_distribution[:, i] ])
                        sentRNN_label = tf.transpose(sentRNN_label)
                        sentRNN_loss = tf.nn.softmax_cross_entropy_with_logits(logits=sentRNN_logistic_mu, labels=sentRNN_label)
                        sentRNN_loss = tf.reduce_sum(sentRNN_loss)/self.batch_size
                        loss += sentRNN_loss * lambda_sent
                        loss_sent += sentRNN_loss
                        

                # wordRNN state
                topic = tf.contrib.rnn.LSTMStateTuple(sent_topic_vec[:, 0:self.wordRNN_lstm_dim], sent_topic_vec[:, self.wordRNN_lstm_dim:])
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

            # tf_vars = {
            #     "loss": loss,
            #     "loss_sent": loss_sent,
            #     "loss_label": loss_label,
            #     "loss_word": loss_word,
            #     "alpha_reg": alpha_reg,
            # }

            
            # if semi == False:
            #     # attention regularization
            #     if self.alpha_c > 0:
            #         sents_mask = tf.expand_dims(sents_mask, 2)
            #         alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2)) * sents_mask  # (N, T, L)
            #         alphas_all = tf.reduce_sum(alphas, 1)      # (N, L)
            #         alpha_reg = self.alpha_c * tf.reduce_sum((self.S_max/4096.0 - alphas_all) ** 2)
            #         tf_vars["loss"] += alpha_reg
            #         tf_vars["alpha_reg"] = alpha_reg

        return loss, loss_sent, loss_word

    
    def build_sampler(self, reuse):

        features = self.densecap_feats # (50, 4096)

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='dense_features', reuse=reuse)

        c, h = self._get_initial_lstm(features=features, reuse=reuse)
        # x = self._word_embedding(inputs=captions_in)
        features_proj = self._project_features(features=features, reuse=reuse)


        # save the generated paragraph to list, here I named generated_sents
        generated_paragraph = []
        total_pred_labels = []
        pred_re = []
        alpha_list = []
        

        # Start build the generation model
        print 'Start build the generation model: '
        for i in range(0, self.S_max):

            if reuse == True:
                tf.get_variable_scope().reuse_variables()
            
            context, alpha = self._attention_layer(features, features_proj, h, reuse=reuse or (i!=0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=reuse or (i!=0))


            with tf.variable_scope('sent_LSTM', reuse=reuse or (i!=0)):
                _, (c, h) = self.sent_LSTM(inputs=context, state=[c, h])

            sent_output = self._decode_lstm(h, context, reuse=reuse or (i!=0))

            with tf.name_scope('fc1'):
                hidden1 = tf.nn.relu( tf.matmul(sent_output, self.fc1_W) + self.fc1_b )
            with tf.name_scope('fc2'):
                sent_topic_vec = tf.nn.relu( tf.matmul(hidden1, self.fc2_W) + self.fc2_b )

            
            sentRNN_logistic_mu = tf.nn.xw_plus_b(sent_output, self.logistic_Theta_W, self.logistic_Theta_b)
            pred = tf.nn.softmax(sentRNN_logistic_mu)
            pred_re.append(pred)

            # save the generated sentence to list, named generated_sent
            generated_sent = []

            # wordRNN state
            topic = tf.contrib.rnn.LSTMStateTuple(sent_topic_vec[:, 0:self.wordRNN_lstm_dim], sent_topic_vec[:, self.wordRNN_lstm_dim:])
            # word_c, word_h = [topic] * self.wordRNN_numlayer
            word_c, word_h = topic
                
            # word RNN, unrolled to N_max time steps
            for j in range(0, self.N_max):
                if j > 0:
                    tf.get_variable_scope().reuse_variables()

                if j == 0:
                    # get word embedding of BOS (index = 0)
                    current_embed = tf.nn.embedding_lookup(self.Wemb, tf.fill([tf.shape(self.densecap_feats)[0]], self.start_idx) )

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





