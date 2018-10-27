import tensorflow as tf
import numpy as np
import time
import os
import hickle
from scipy import ndimage
from util import *
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__))) # for batch inference
from evaluate import evaluate
from data_loader import SemiTrainingData, TrainingData
import tensorflow.contrib.slim as slim
# from bleu import evaluate

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True


class ParagraphSolver(object):
    def __init__(self, model, data, config, opts):

        self.save_every = 20
        self.save_every_aftere350epoch = 5
        self.early_stop_epoch = 10
        self.early_stop = opts.early_stop
        self.ref_test_sents = opts.ref_test_sents

        self.config = config
        self.model = model
        self.process_name = opts.process_name
        self.pretrained_model = opts.model_name
        self.data = data
        self.fixed_n_sent = opts.fixed_n_sent
        self.transfered_model_name = opts.transfered_model_name
        self.semi_training = opts.semi_training

        self.T_stop = config.T_stop
        self.n_epoch = config.n_epoch
        self.batch_size = config.batch_size
        self.test_batch_size = config.test_batch_size
        self.learning_rate = config.learning_rate
        self.log_path = os.path.join(config.log_dir, self.process_name) 
        self.model_path = os.path.join(config.model_dir, self.process_name)
        self.result_path = os.path.join(config.result_dir, self.process_name)
        self.update_rule = opts.update_rule


        if self.semi_training:
            self.semi_dense_feats_files = config.semi_dense_feats_files
            self.semi_captions_files = config.semi_captions_files

        # get pretrained_epoch
        if self.pretrained_model is not None:
            self.pretrained_epoch = int(self.pretrained_model.split('-')[1] )
            self.log_file = "log_" + self.pretrained_model + ".txt"
            self.score_file = "score_" + self.pretrained_model + ".txt"
        else:
            self.pretrained_epoch = 0
            self.log_file = "log.txt" 
            self.score_file = "score.txt"
        

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


    def init_session(self, max_to_keep=20):
        
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
            grads_and_vars = list(zip(clipped_gradients, params))
            train_op = optimizer.apply_gradients( grads_and_vars=grads_and_vars )

        return train_op, grads_and_vars

   
    def model_summary(self):
        print "-" * 50 + '\n'
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)
        print "-" * 50 + '\n'

    def run_epoch(self, sess, train_data, loss_dict, tf_vars, summary_op, summary_writer, e, semi=False):

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
            
                _, _loss, _loss_sent, _loss_label, _loss_word, _alpha_reg = sess.run(
                    [tf_vars["train_op"], 
                     tf_vars["loss"], 
                     tf_vars["loss_sent"], 
                     tf_vars["loss_label"], 
                     tf_vars["loss_word"],
                     tf_vars["alpha_reg"]], feed_dict)


                loss_dict["total_sent_loss"] += _loss_sent
                loss_dict["alpha_reg"] += _alpha_reg

            loss_dict["total_loss"] += _loss
            loss_dict["total_label_loss"] += _loss_label
            loss_dict["total_word_loss"] += _loss_word
        

        # average    
        for key in loss_dict.keys():
            loss_dict[key] = loss_dict[key] / train_data.num_batch

        summary = sess.run(summary_op, feed_dict)
        summary_writer.add_summary(summary, e)

        return loss_dict

    def train(self):
        
        # build graphs for training model and sampling captions
        # This scope fixed things!!
        with tf.variable_scope(tf.get_variable_scope()):
            _tf_vars = self.model.build_model(S_max=6)
            sampled_paragraphs, pred_re = self.model.build_sampler(reuse=True)
        
        tf_vars = {}
        for key, value in _tf_vars.iteritems():
            tf_vars[key] = value

        train_op, grads_and_vars = self.backprop(tf_vars["loss"])

        tf_vars["sampled_paragraphs"] = sampled_paragraphs
        tf_vars["pred_re"] = pred_re
        tf_vars["train_op"] = train_op

        if self.semi_training:

            _semi_tf_vars = self.model.build_model(S_max=1, semi=True, reuse=True)
            
            for key, value in _semi_tf_vars.iteritems():
                s_key = 's_' + key
                tf_vars[s_key] = value

            s_train_op = self.backprop(tf_vars["s_loss"], semi=True, reuse=True)
            tf_vars["s_train_op"] = s_train_op


        # summary visualization
        loss_scalars = [(key, tf_vars[key]) for key in tf_vars.keys() if 'loss' in key]
        for (key, tf_var) in loss_scalars:
            tf.summary.scalar(key, tf_var)       

        for grad, var in grads_and_vars:
            try:
                tf.summary.histogram(var.op.name, var)
                tf.summary.histogram(var.op.name+'/gradient', grad)
            except:
                print var.op.name

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())


        # init session
        sess, saver = self.init_session()
        
        self.model_summary()

        # for early stop (METEOR + CIDEr)
        threshold = 0 
        threshold_no_change_epoch = 0

        train_data = self.data.train_data

        # start training
        start_t = time.time()
        scores = []
        

        with open(os.path.join(self.log_path, self.log_file), 'w') as log:
            with open(os.path.join(self.result_path, self.score_file), 'w') as f_score:
            
                print "start training from %d epoch" % self.pretrained_epoch
                for e in range(self.n_epoch):

                    # skip epoch
                    if e < self.pretrained_epoch:
                        continue

                    loss_dict = {
                        "total_loss": 0.0,
                        "total_label_loss": 0.0,
                        "total_sent_loss": 0.0,
                        "total_word_loss": 0.0,
                        "alpha_reg": 0.0,
                    }


                    # semi training
                    # but data is so big that cannot load all in once, so i split.
                    if self.semi_training:
                        if (e+1) % 5 == 0:
                            for s_feats_files, s_captions_files in zip(self.semi_dense_feats_files, self.semi_captions_files):
                                print s_feats_files + " is loading..."
                                train_data = SemiTrainingData(s_feats_files, s_captions_files, self.batch_size)
                                self.run_epoch(sess, train_data, loss_dict, tf_vars, summary_op, summary_writer, e+1, semi=True)
                           
                            # training img to paragraph
                            print "im2p is loading..."
                            train_data = TrainingData(self.config, self.batch_size)
                            self.run_epoch(sess, train_data, loss_dict, tf_vars, summary_op, summary_writer, e+1)

                        else:
                            self.run_epoch(sess, train_data, loss_dict, tf_vars, summary_op, summary_writer, e+1)

                    else:
                        self.run_epoch(sess, train_data, loss_dict, tf_vars, summary_op, summary_writer, e+1)


                    # print loss 
                    msg1 = 'Epoch: %d, loss: %f, loss_sent: %f, loss_label: %f, loss_word: %f, alpha_reg: %f, Time cost: %f' % \
                          (e+1, loss_dict["total_loss"], loss_dict["total_sent_loss"], loss_dict["total_label_loss"], loss_dict["total_word_loss"], loss_dict["alpha_reg"], time.time() - start_t)
                    print msg1
                    log.write(msg1 + '\n')


                    # early stopping
                    # if threshold < (final_scores['METEOR'] + final_scores['CIDEr']):
                    #     threshold = final_scores['METEOR'] + final_scores['CIDEr']
                    #     threshold_no_change_epoch = 0
                    # else:
                    #     threshold_no_change_epoch += 1

                    # if self.early_stop == True:
                    #     if threshold_no_change_epoch >= self.early_stop_epoch:
                    #         saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e+1)
                    #         print "model-%s saved." % (e+1)
                    #         break

                    # save model's parameters and validate
                    if ((e+1) % self.save_every == 0 and (e+1) < 350) or \
                       ((e+1) % self.save_every_aftere350epoch == 0 and (e+1) >= 350):  
                        
                        # validate
                        output_path = os.path.join( self.result_path, "val_candidate_" + str(e+1) + "_txt")
                        self.validate(sess, sampled_paragraphs, pred_re, output_path)

                        # print evaluation score
                        final_scores = evaluate(get_scores=True, reference_path="../data/reference.txt", candidate_path=output_path)
                        msg2 = "epoch: %d ==> Bleu_1: %f, Bleu_2: %f, Bleu_3: %f, Bleu_4: %f, METEOR: %f, CIDEr: %f" \
                            % (e+1, final_scores['Bleu_1'], final_scores['Bleu_2'], final_scores['Bleu_3'],
                            final_scores['Bleu_4'], final_scores['METEOR'], final_scores['CIDEr'])
                        print msg2
                        log.write(msg2 + '\n')
                        print "-"*50

                        f_score.write(msg2)

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

        
        if self.ref_test_sents == True:
            n_test_data_sents_path = self.config.n_test_data_sents_path 
            n_test_data_sents = hickle.load(n_test_data_sents_path)

        for i in xrange(test_data.num_batch):
            batch_data = test_data.next_batch()

            feed_dict = {
                 self.model.densecap_feats: batch_data["densecap_feats"],
            }

            _sampled_paragraphs, _pred = sess.run([sampled_paragraphs, pred_re], feed_dict)
            
            
            infer_paragraphs = decode_paragraphs(
                _sampled_paragraphs, _pred, idx2word, 
                fixed_n_sent=self.fixed_n_sent, 
                n_test_data_sents=n_test_data_sents[i*self.test_batch_size:(i+1)*self.test_batch_size] if self.ref_test_sents==True else None)


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




