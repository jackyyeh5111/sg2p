import tensorflow as tf
import numpy as np
import time
import os
import pickle as pickle
from scipy import ndimage
from util import *
import sys
# print 'cwd (before):', os.getcwd()
os.chdir('/2t/jackyyeh/im2p/core') # for batch inference
from evaluate import evaluate
# from bleu import evaluate

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True


class ParagraphSolver(object):
    def __init__(self, model, data, config, opts):

        self.save_every = 20
        self.early_stop_epoch = 10
        self.early_stop = opts.early_stop


        self.model = model
        self.process_name = opts.process_name
        self.pretrained_model = opts.model_name
        self.data = data
        self.fixed_n_sent = opts.fixed_n_sent
        
        self.T_stop = config.T_stop
        self.n_epoch = config.n_epoch
        self.batch_size = config.batch_size
        self.test_batch_size = config.test_batch_size
        self.learning_rate = config.learning_rate
        self.log_path = os.path.join(config.log_dir, self.process_name) 
        self.model_path = os.path.join(config.model_dir, self.process_name)
        self.result_path = os.path.join(config.result_dir, self.process_name)
        self.update_rule = config.update_rule

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)


    def init_session(self, max_to_keep=None):
        
        sess = tf.Session(config=tf_config)
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=max_to_keep)

        if self.pretrained_model is not None:
            print "Start training with pretrained Model.."
            saver.restore(sess, os.path.join(self.model_path, self.pretrained_model) )
                    
        return sess, saver


    def validate(self, sess, sampled_paragraphs, pred_re, output_path):
        val_data = self.data.val_data

        # validation per epoch
        val_data.reset_pointer()

        totol_paragraphs = []
        for val_b in xrange(val_data.num_batch):
            batch_data = val_data.next_batch()
            feed_dict = {
                     self.model.densecap_feats: batch_data["densecap_feats"]
                }
            _sampled_paragraphs, _pred = sess.run([sampled_paragraphs, pred_re], feed_dict)
            val_paragraphs = decode_paragraphs(_sampled_paragraphs, _pred, self.data.idx2word, fixed_n_sent=self.fixed_n_sent)
            totol_paragraphs.extend(val_paragraphs)
            
        output_paragraphs(totol_paragraphs, output_path)


    def train(self):
        
        # build graphs for training model and sampling captions
        # This scope fixed things!!
        with tf.variable_scope(tf.get_variable_scope()):
            loss, loss_sent, loss_label, loss_word = self.model.build_model()
            sampled_paragraphs, pred_re = self.model.build_sampler(reuse=True)
    
        # train op
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            
            max_gradient_norm=1.0

            params = tf.trainable_variables()
            gradients = tf.gradients(loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm( gradients, max_gradient_norm )

            # Optimization
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            train_op = optimizer.apply_gradients( zip(clipped_gradients, params) )

            # optimizer = self.optimizer(learning_rate=self.learning_rate)
            # grads = tf.gradients(loss, tf.trainable_variables())
            # grads_and_vars = list(zip(grads, tf.trainable_variables()))
            # train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)


        train_data = self.data.train_data
        num_batch = train_data.num_batch
       

        # summary op
        # tf.scalar_summary('batch_loss', loss)
        # tf.summary.scalar('batch_loss', loss)
        # tf.summary.scalar('batch_loss_label', loss_label)
        # tf.summary.scalar('batch_loss_word', loss_word)
        # for var in tf.trainable_variables():
        #     #tf.histogram_summary(var.op.name, var)
        #     tf.summary.histogram(var.name, var)
        # for grad, var in grads_and_vars:
        #     #tf.histogram_summary(var.op.name+'/gradient', grad)
        #     try:
        #         tf.summary.histogram(var.name+'/gradient', grad)
        #     except:
        #         print var.name

        #summary_op = tf.merge_all_summaries()
        # summary_op = tf.summary.merge_all()

        print "The number of epoch: %d" % self.n_epoch
        print "Data size: %d" % train_data.size
        print "Batch size: %d" % self.batch_size
        print "Iterations per epoch: %d" % num_batch
        
        # init session
        sess, saver = self.init_session()
        # summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
        
        pretrained_epoch = 0
        if self.pretrained_model is not None:
            pretrained_epoch = int(self.pretrained_model.split('-')[1] )
            print "start training from %d epoch" % pretrained_epoch

        
        threshold = 0 # for early stop (METEOR + CIDEr)
        threshold_no_change_epoch = 0
        
        # start training
        start_t = time.time()
        log_file = "log.txt"
        if self.pretrained_model is not None:
            log_file = "log_" + self.pretrained_model + ".txt"

        with open(os.path.join(self.log_path, log_file), 'w') as log:
        
            for e in range(self.n_epoch):

                if e < pretrained_epoch:
                    continue

                train_data.reset_pointer()

                total_loss = 0.0
                total_label_loss = 0.0
                total_sent_loss = 0.0
                total_word_loss = 0.0
                for i in range(num_batch):
                    batch_data = train_data.next_batch()

                    feed_dict = {
                         self.model.densecap_feats: batch_data["densecap_feats"],
                         self.model.num_distribution: batch_data["num_distribution"],
                         self.model.captions: batch_data["captions"],
                         # self.model.caption_labels: batch_data["caption_labels"],
                    }
                    
                    _, _loss, _loss_sent, _loss_label, _loss_word = sess.run([train_op, loss, loss_sent, loss_label, loss_word], feed_dict)
        
                    total_loss += _loss
                    total_label_loss += _loss_label
                    total_word_loss += _loss_word
                    total_sent_loss += _loss_sent

                    # write summary for tensorboard visualization
                    # if i % 10 == 0:
                    #     summary = sess.run(summary_op, feed_dict)
                    #     summary_writer.add_summary(summary, e*num_batch + i)


                # monitor
                msg1 = 'Epoch: %d, loss: %f, loss_sent: %f, loss_label: %f, loss_word: %f, Time cost: %f' % \
                      (e+1, total_loss/num_batch, total_sent_loss/num_batch, total_label_loss/num_batch, total_word_loss/num_batch, time.time() - start_t)
                print msg1
                log.write(msg1 + '\n')

                # validate
                output_path = os.path.join( self.result_path, "val_candidate.txt" )
                self.validate(sess, sampled_paragraphs, pred_re, output_path)

                # print evaluation score
                final_scores = evaluate(get_scores=True, reference_path="../data/val_reference.txt", candidate_path=output_path)
                msg2 = "epoch: %d ==> Bleu_1: %f, Bleu_2: %f, Bleu_3: %f, Bleu_4: %f, METEOR: %f, CIDEr: %f" \
                    % (e+1, final_scores['Bleu_1'], final_scores['Bleu_2'], final_scores['Bleu_3'],
                    final_scores['Bleu_4'], final_scores['METEOR'], final_scores['CIDEr'])
                print msg2
                log.write(msg2 + '\n')

                print "-"*50



                # early stopping
                if threshold < (final_scores['METEOR'] + final_scores['CIDEr']):
                    threshold = final_scores['METEOR'] + final_scores['CIDEr']
                    threshold_no_change_epoch = 0
                else:
                    threshold_no_change_epoch += 1

                if self.early_stop == True:
                    if threshold_no_change_epoch >= self.early_stop_epoch:
                        saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e+1)
                        print "model-%s saved." % (e+1)
                        break

                # save model's parameters and validate
                if (e+1) % self.save_every == 0:  
                    # save model
                    saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e+1)
                    print "model-%s saved." % (e+1)



    def inference(self):

        idx2word = self.data.idx2word
        test_data = self.data.test_data

        with tf.variable_scope(tf.get_variable_scope()):
            sampled_paragraphs, pred_re = self.model.build_sampler(reuse=False)

        sess, _ = self.init_session()

        # start inference
        start_time = time.time()        
        print "start inference"
        totol_paragraphs = []
        infered_paragraph = 0
        total_max_pred_words = []
        for i in xrange(test_data.num_batch):
            batch_data = test_data.next_batch()

            feed_dict = {
                 self.model.densecap_feats: batch_data["densecap_feats"],
            }

            _sampled_paragraphs, _pred = sess.run([sampled_paragraphs, pred_re], feed_dict)
            
            infer_paragraphs = decode_paragraphs(_sampled_paragraphs, _pred, idx2word, fixed_n_sent=self.fixed_n_sent)
            totol_paragraphs.extend(infer_paragraphs)

            # batch_max_pred_words = get_max_pred_words(_total_pred_labels, 20, idx2word)
            # total_max_pred_words.extend(batch_max_pred_words)
            # for i in xrange(10):
            #     print batch_max_pred_words[i]
                
            # raw_input()
            infered_paragraph += len(batch_data["densecap_feats"])
            print "%d paragraph have been infered" % infered_paragraph

        if self.fixed_n_sent:
            output_path = os.path.join( self.result_path, self.pretrained_model + "_fixed_n_sent")
            output_score_path = os.path.join( self.result_path, 'score' + "_fixed_n_sent")
        else:
            output_path = os.path.join( self.result_path, self.pretrained_model)
            output_score_path = os.path.join( self.result_path, 'score')

        output_paragraphs(totol_paragraphs, output_path)
        # output_path_max_pred_words = os.path.join( self.result_path, self.pretrained_model + "_pred_words")
        # output_max_pred_words(total_max_pred_words, output_path_max_pred_words) 

        final_scores = evaluate(get_scores=True, reference_path="../data/reference.txt", candidate_path=output_path)

        # output score
        with open(output_score_path, 'a') as f:
            model_name = int(self.pretrained_model.split('-')[1])
            msg2 = "model: %d ==> Bleu_1: %.4f, Bleu_2: %.4f, Bleu_3: %.4f, Bleu_4: %.4f, METEOR: %.4f, CIDEr: %.4f" \
                        % (model_name, final_scores['Bleu_1'], final_scores['Bleu_2'], final_scores['Bleu_3'],
                        final_scores['Bleu_4'], final_scores['METEOR'], final_scores['CIDEr'])
            print msg2
            f.write(msg2 + '\n')

        

        print "Time cost: " + str(time.time()-start_time)




