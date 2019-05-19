import tensorflow as tf
import numpy as np
import time
import os
# import hickle
# from scipy import ndimage
from util import *
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__))) # for batch inference
from evaluate import evaluate
from data_loader import TrainingData
from tqdm import tqdm
import pickle
import tensorflow.contrib.slim as slim
# from bleu import evaluate

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True


class ParagraphSolver(object):
    def __init__(self, model, dc, args):

        self.args = args

        self.save_every = self.args.save_every
        self.log_every = self.args.log_every
        
        
        self.ref_test_sents = self.args.ref_test_sents

        self.model = model
        self.process_name = self.args.process_name
        self.pretrained_model = self.args.model_name
        self.dc = dc
        self.fixed_n_sent = self.args.fixed_n_sent
        self.use_attrs = self.args.use_attrs
        self.spt_feats = self.args.spt_feats

        self.n_epoch = self.args.n_epochs
        self.batch_size = self.args.batch_size
        self.test_batch_size = self.args.test_batch_size
        self.use_box_feats = self.args.use_box_feats

        self.learning_rate = self.args.learning_rate
        self.log_path = os.path.join(self.args.path.log_dir, self.process_name) 
        self.model_path = os.path.join(self.args.path.model_dir, self.process_name)
        self.result_path = os.path.join(self.args.path.result_dir, self.process_name)
        self.update_rule = self.args.update_rule


        # for early stop
        self.patience = self.args.patience
        self.best_score = 0
        self.best_epoch = 0
        self.score_no_change = 0
        self.is_early_stop = self.args.is_early_stop
        

        # get pretrained_epoch
        if self.pretrained_model is not None:
            self.pretrained_epoch = int(self.pretrained_model.split('-')[1] )
            self.open_file_mode = 'a'
            # self.log_file = "log_" + self.pretrained_model + ".txt"
            # self.score_file = "score_" + self.pretrained_model + ".txt"
        else:
            self.pretrained_epoch = 0
            self.open_file_mode = 'w'
            # self.log_file = "log.txt" 
            # self.score_file = "score.txt"
        

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


    def init_session(self):
        
        sess = tf.Session(config=tf_config)
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()

        if self.pretrained_model is not None:
            print "Start training with pretrained Model.."
            saver.restore(sess, os.path.join(self.model_path, self.pretrained_model) )
                    
        return sess, saver

    def backprop(self, loss, semi=False, reuse=False, max_gradient_norm=1.0):
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            # Calculate and clip gradients
            params = tf.trainable_variables()

            gradients = tf.gradients(loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm( gradients, max_gradient_norm )

            # Optimization
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            grads_and_vars = list(zip(clipped_gradients, params))
            train_op = optimizer.apply_gradients( grads_and_vars=grads_and_vars )

        return train_op, grads_and_vars

   
    
    def run_epoch(self, train_op, loss, loss_sent, loss_word, epoch, semi=False):
        
        train_data = self.dc.train_data
        train_data.reset_pointer()

        total_loss = 0
        total_sent_loss = 0
        total_word_loss = 0

        total_step = train_data.num_batch

        for i in tqdm(range(train_data.num_batch)):
            batch_data = train_data.next_batch()

        
            feed_dict = {
                 # self.model.densecap_feats: batch_data["densecap_feats"],
                 self.model.objs: batch_data["objs"],
                 self.model.triples: batch_data["triples"],
                 self.model.num_distribution: batch_data["num_distribution"],
                 self.model.captions: batch_data["captions"],
            }

            if self.use_box_feats:
                feed_dict[self.model.box_feats] = batch_data["box_feats"]
                if self.spt_feats:
                    feed_dict[self.model.rel_feats] = batch_data["rel_feats"]
            
            
            if self.use_attrs:
                feed_dict[self.model.attrs] = batch_data["attrs"]

            
            _, _loss, _loss_sent, _loss_word = self.sess.run(
                [train_op, 
                 loss, 
                 loss_sent,             
                 loss_word], feed_dict)


            total_sent_loss += _loss_sent
            total_loss += _loss
            total_word_loss += _loss_word
        

        summary = self.sess.run(self.summary_op, feed_dict)
        self.summary_writer.add_summary(summary, epoch)

        return total_loss/total_step, total_sent_loss/total_step, total_word_loss/total_step

    def _summary(self, grads_and_vars, **kwargs):
        '''
            for tensorboard
        '''
        for grad, var in grads_and_vars:
            try:
                tf.summary.histogram(var.op.name, var)
                tf.summary.histogram(var.op.name+'/gradient', grad)
            except:
                print (var.op.name)

        for key in kwargs:
            tf.summary.scalar('key', kwargs[key])

        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())

    def _print_model_vars(self):
        print ("-" * 50 + '\n')
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)
        print ("-" * 50 + '\n')


    def _print_scores(self, final_scores, epoch, f_score):
        msg = ("epoch: %d ==> Bleu_1: %f, Bleu_2: %f, Bleu_3: %f, Bleu_4: %f, METEOR: %f, CIDEr: %f" 
                % (epoch+1, final_scores['Bleu_1'], final_scores['Bleu_2'], final_scores['Bleu_3'],
                final_scores['Bleu_4'], final_scores['METEOR'], final_scores['CIDEr']))

        print (msg)        
        f_score.write(msg + '\n')


    def _print_logs(self, total_loss, sent_loss, word_loss, f_log, start_t, epoch):

        msg = ('Epoch: %d, loss: %f, loss_sent: %f, loss_word: %f, Time cost: %f' % 
                (epoch+1, total_loss, sent_loss, word_loss, time.time() - start_t))
        
        print (msg)
        f_log.write(msg + '\n')

    def _run_validate(self, sampled_paragraphs, pred_re, alphas, output_path, temp_obj_vecs, mode='val'):
        
        assert mode in ['val', 'test']

        if mode == 'val':
            val_data = self.dc.val_data
        elif mode == 'test':
            val_data = self.dc.test_data

        # validation per epoch
        val_data.reset_pointer()

        totol_paragraphs = []
        output_alphas = np.zeros((2489, 6, 250))
        for i in tqdm(range(val_data.num_batch)):
           
            batch_data = val_data.next_batch()
            feed_dict = {
                    self.model.objs: batch_data["objs"],
                    self.model.triples: batch_data["triples"],
                }

            if self.use_box_feats:
                feed_dict[self.model.box_feats] = batch_data["box_feats"]
                if self.spt_feats:
                    feed_dict[self.model.rel_feats] = batch_data["rel_feats"]


            if self.use_attrs:
                feed_dict[self.model.attrs] = batch_data["attrs"]

            
            _temp_obj_vecs, _sampled_paragraphs, _pred, _alphas = self.sess.run([temp_obj_vecs, sampled_paragraphs, pred_re, alphas], feed_dict)
            
            # print _temp_obj_vecs.shape
            # print _temp_obj_vecs[0]
            # print '-' * 5 + '\n'
            # print _temp_obj_vecs[0][0]
            # print '-' * 5 + '\n'
            # print _temp_obj_vecs[0][140]
            # raw_input()

            print ("_alphas.shape", _alphas.shape)
            output_alphas[i*256:i*256+len(_alphas)] = _alphas

            val_paragraphs = decode_paragraphs(_sampled_paragraphs, _pred, self.dc.idx2word, fixed_n_sent=self.fixed_n_sent)
            totol_paragraphs.extend(val_paragraphs)
            
        output_paragraphs(totol_paragraphs, output_path)

        alpha_path = os.path.join( self.result_path, "alphas.pkl")
        with open(alpha_path, 'w') as f:
            pickle.dump(output_alphas, f)


    def _early_stop(self, current_score, epoch):

        if self.best_score > current_score:
            self.score_no_change += 1
            if self.score_no_change == self.patience:
                return True
        else:
            self.best_score = current_score
            self.score_no_change = 0
            self.best_epoch = epoch
        
        return False
        

    def train(self):
        
        with tf.variable_scope(tf.get_variable_scope()):
            loss, loss_sent, loss_word = self.model.build_model(S_max=6)
            sampled_paragraphs, pred_re = self.model.build_sampler(reuse=True)
        
        train_op, grads_and_vars = self.backprop(loss)

        self._summary(grads_and_vars, loss=loss, loss_sent=loss_sent, loss_word=loss_word)

        # init session
        self.sess, self.saver = self.init_session()
        # self._print_model_vars()

        # start training
        start_t = time.time()

        f_log = open(os.path.join(self.log_path, 'log.txt'), self.open_file_mode, buffering=0)
        f_score = open(os.path.join(self.result_path, 'score.txt'), self.open_file_mode, buffering=0)
    
        print ("start training from %d epoch" % self.pretrained_epoch)
        for epoch in range(self.n_epoch):
            
            # skip epoch
            if epoch < self.pretrained_epoch:
                continue

            total_loss, total_sent_loss, total_word_loss = self.run_epoch(train_op, loss, loss_sent, loss_word, epoch+1)


            self._print_logs( total_loss,
                      total_sent_loss, 
                      total_word_loss, 
                      f_log, start_t, epoch)


            if (epoch+1) % self.log_every == 0:

                output_path = os.path.join( self.result_path, "val_candidate_" + str(epoch+1) + ".txt")
                self._run_validate(sampled_paragraphs, pred_re, output_path)
                final_scores = evaluate(get_scores=True, reference_path=self.args.path.reference_path, candidate_path=output_path)
                self._print_scores(final_scores, epoch, f_score)

                # early stop depends on score 'METEOR', 'CIDEr'
                if self.is_early_stop == True:
                    es_score = final_scores['METEOR'] + final_scores['CIDEr']
                    if self._early_stop(es_score, epoch) == True:
                        msg = ('early stop in epoch[%d]!\nbest_score: %.2f' % (self.best_epoch, self.best_score))
                        print (msg)
                        f_score.write(msg + '\n')
                        break

            if (epoch+1) % self.save_every == 0:
                self._save_model(epoch)


        f_log.close()
        f_score.close()

    def _save_model(self, epoch):
        self.saver.save(self.sess, os.path.join(self.model_path, 'model'), global_step=epoch+1)
        print "model-%s saved." % (epoch+1)



    def inference(self):

        with tf.variable_scope(tf.get_variable_scope()):
            sampled_paragraphs, pred_re, alphas, temp_obj_vecs = self.model.build_sampler(reuse=False)

        self.sess, _ = self.init_session()

        # start inference
        start_time = time.time()        
        print "--- start inference ---"
        totol_paragraphs = []
        infered_paragraph = 0
        total_max_pred_words = []

        epoch = int(self.pretrained_model.split('-')[-1])

        # f_score = open(os.path.join(self.result_path, 'score_modify.txt'), 'a', buffering=0)
        output_path = '/2t/jackyyeh/neural-motifs/vis/data/demo/result_demo.txt'
        # output_path = os.path.join( self.result_path, "infer_" + str(epoch) + ".txt")
        self._run_validate(sampled_paragraphs, pred_re, alphas, output_path, temp_obj_vecs, mode='test')
        # final_scores = evaluate(get_scores=True, reference_path=self.args.path.reference_path, candidate_path=output_path)
        
        # msg = ("epoch: %d ==> Bleu_1: %f, Bleu_2: %f, Bleu_3: %f, Bleu_4: %f, METEOR: %f, CIDEr: %f" 
        #         % (epoch-1, final_scores['Bleu_1'], final_scores['Bleu_2'], final_scores['Bleu_3'],
        #         final_scores['Bleu_4'], final_scores['METEOR'], final_scores['CIDEr']))

        # print (msg)        
        # f_score.write(msg + '\n')

        # self._print_scores(final_scores, epoch-1, f_score)
