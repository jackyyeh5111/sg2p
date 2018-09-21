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
from data_loader import SemiTrainingData, TrainingData
import tensorflow.contrib.slim as slim
# from bleu import evaluate

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True


class ParagraphSolver(object):
    def __init__(self, model, data, config, opts):

        self.save_every = 9
        self.early_stop_epoch = 10
        self.early_stop = opts.early_stop

        self.config = config
        self.model = model
        self.process_name = opts.process_name
        self.pretrained_model = opts.model_name
        self.data = data
        self.fixed_n_sent = opts.fixed_n_sent
        self.semi_dense_feats_files = config.semi_dense_feats_files
        self.semi_captions_files = config.semi_captions_files
        self.transfered_model_name = opts.transfered_model_name
        
        self.T_stop = config.T_stop
        self.n_epoch = config.n_epoch
        self.batch_size = config.batch_size
        self.test_batch_size = config.test_batch_size
        self.learning_rate = config.learning_rate
        self.log_path = os.path.join(config.log_dir, self.process_name) 
        self.model_path = os.path.join(config.model_dir, self.process_name)
        self.result_path = os.path.join(config.result_dir, self.process_name)
        self.update_rule = config.update_rule


        # get pretrained_epoch
        if self.pretrained_model is not None:
            self.pretrained_epoch = int(self.pretrained_model.split('-')[1] )
            self.log_file = "log_" + self.pretrained_model + ".txt"
        else:
            self.pretrained_epoch = 0
            self.log_file = "log.txt" 
        

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
                    

        elif self.transfered_model_name is not None:

            variables = slim.get_variables_to_restore()
            variables_to_restore = [v for v in variables \
                if "label" not in v.name 
                and "logistic_Theta" not in v.name
                and "beta" not in v.name ] 
            
            saver_restore = tf.train.Saver(variables_to_restore)

            print "Start training with pretrained Model.."
            saver_restore.restore(sess, self.transfered_model_name )
            
            saver = tf.train.Saver(max_to_keep=max_to_keep) 

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


    def backprop(self, loss, semi=False, reuse=False, max_gradient_norm=1.0):
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            # Calculate and clip gradients
            if semi == True:
                params = [var for var in tf.trainable_variables() if "word_LSTM" in var.name ]
            else:
                params = tf.trainable_variables()

            gradients = tf.gradients(loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm( gradients, max_gradient_norm )

            # Optimization
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            train_op = optimizer.apply_gradients( zip(clipped_gradients, params) )

        return train_op

    def summary_scalars(self, tf_vars):
        tf.summary.scalar('loss', tf_vars["loss"])
        tf.summary.scalar('loss_sent', tf_vars["loss_sent"])
        tf.summary.scalar('loss_word', tf_vars["loss_word"])
        tf.summary.scalar('loss_label', tf_vars["loss_label"])
        tf.summary.scalar('s_loss', tf_vars["s_loss"])
        tf.summary.scalar('s_loss_word', tf_vars["s_loss_word"])
        

    def model_summary(self):
        print "-" * 50 + '\n'
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)
        print "-" * 50 + '\n'

    def run_epoch(self, sess, train_data, loss_dict, tf_vars, semi=False):

        train_data.reset_pointer()

        for i in range(train_data.num_batch):
            batch_data = train_data.next_batch()

            if semi == True:
                feed_dict = {
                     self.model.densecap_feats: batch_data["densecap_feats"],
                     self.model.coco_captions: batch_data["captions"],
                     # self.model.caption_labels: batch_data["caption_labels"],
                }

                _, _loss, _loss_label, _loss_word = sess.run(
                    [tf_vars["s_train_op"], 
                     tf_vars["s_loss"], 
                     tf_vars["s_loss_label"], 
                     tf_vars["s_loss_word"]], feed_dict)

            else:
                feed_dict = {
                     self.model.densecap_feats: batch_data["densecap_feats"],
                     self.model.num_distribution: batch_data["num_distribution"],
                     self.model.captions: batch_data["captions"],
                     # self.model.caption_labels: batch_data["caption_labels"],
                }
            
                _, _loss, _loss_sent, _loss_label, _loss_word = sess.run(
                    [tf_vars["train_op"], 
                     tf_vars["loss"], 
                     tf_vars["loss_sent"], 
                     tf_vars["loss_label"], 
                     tf_vars["loss_word"]], feed_dict)


                loss_dict["total_sent_loss"] += _loss_sent

            loss_dict["total_loss"] += _loss
            loss_dict["total_label_loss"] += _loss_label
            loss_dict["total_word_loss"] += _loss_word
            
        return loss_dict

    def train(self):
        
        # build graphs for training model and sampling captions
        # This scope fixed things!!
        with tf.variable_scope(tf.get_variable_scope()):
            loss, loss_sent, loss_label, loss_word = self.model.build_model(S_max=6)
            s_loss, _, s_loss_label, s_loss_word = self.model.build_model(S_max=1, semi=True, reuse=True)
            sampled_paragraphs, pred_re = self.model.build_sampler(reuse=True)
    
        train_op = self.backprop(loss)
        s_train_op = self.backprop(s_loss, semi=True, reuse=True)
        
        tf_vars = {
            "loss": loss,
            "loss_sent": loss_sent,
            "loss_label": loss_label,
            "loss_word": loss_word,
            "s_loss": s_loss,
            "s_loss_word": s_loss_word,
            "s_loss_label": s_loss_label,
            "sampled_paragraphs": sampled_paragraphs,
            "pred_re": pred_re,
            "train_op": train_op,
            "s_train_op": s_train_op,
        }

        # summary 
        self.summary_scalars(tf_vars)
        summary_op = tf.summary.merge_all()

        # init session
        sess, saver = self.init_session()
        summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
        self.model_summary()

        # for early stop (METEOR + CIDEr)
        threshold = 0 
        threshold_no_change_epoch = 0

        train_data = self.data.train_data

        # start training
        start_t = time.time()
        with open(os.path.join(self.log_path, self.log_file), 'w') as log:
            
            print "start training from %d epoch" % self.pretrained_epoch
            for e in range(self.n_epoch):

                # skip epoch
                if e < self.pretrained_epoch:
                    continue

                loss_dict = {
                    "total_loss": 0.0,
                    "total_label_loss": 0.0,
                    "total_sent_loss": 0.0,
                    "total_word_loss": 0.0
                }

                # semi training
                # but data is so big that cannot load all in once, so i split.
                if (e+1) % 5 == 0:
                    for s_feats_files, s_captions_files in  zip(self.semi_dense_feats_files, self.semi_captions_files):
                        print s_feats_files + " is loading..."
                        train_data = SemiTrainingData(s_feats_files, s_captions_files, self.batch_size)
                        self.run_epoch(sess, train_data, loss_dict, tf_vars, semi=True)
                   
                    # training img to paragraph
                    print "im2p is loading..."
                    train_data = TrainingData(self.config, self.batch_size)
                    self.run_epoch(sess, train_data, loss_dict, tf_vars)

                else:
                    self.run_epoch(sess, train_data, loss_dict, tf_vars)

                # write summary for tensorboard visualization
                # if e % 10 == 0:
                # summary = sess.run(summary_op, feed_dict)
                # summary_writer.add_summary(summary, e)

                # print loss 
                msg1 = 'Epoch: %d, loss: %f, loss_sent: %f, loss_label: %f, loss_word: %f, Time cost: %f' % \
                      (e+1, loss_dict["total_loss"], loss_dict["total_sent_loss"], loss_dict["total_label_loss"], loss_dict["total_word_loss"], time.time() - start_t)
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




