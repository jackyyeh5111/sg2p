import numpy as np
import os
import random

# others
SEED = 55
random.seed(SEED)
np.random.seed(SEED)


class Config():
	
	num_boxes = 50 # number of Detected regions in each image
	feats_dim = 4096 # feature dimensions of each regions
	project_dim = 1024 # project the features to one vector, which is 1024 dimensions

	sentRNN_lstm_dim = 512 # the sentence LSTM hidden units
	sentRNN_FC_dim = 1024 # the fully connected units
	wordRNN_lstm_dim = 512 # the word LSTM hidden units
	word_embed_dim = 1024 # the learned embedding vectors for the words
	sentRNN_numlayer = 1
	wordRNN_numlayer = 2

	S_max = 6
	N_max = 30
	T_stop = 0.5
	n_epoch = 800

	update_rule = 'adam'
	learning_rate = 0.0001
	batch_size = 128
	test_batch_size = 256


	# data path
	log_dir = '../logs/'
	model_dir = "../models/"
	result_dir = "../results/"

	train_feats_path = '../data/densecap/im2p_train_output.hkl'
	validate_feats_path = '../data/densecap/im2p_val_output.hkl'
	test_feats_path = '../data/densecap/im2p_test_output.hkl'

	attrs_path = '../attrs.txt'
	
	word2index_path = '../data/word2idx.json'
	index2word_path = '../data/idx2word.json'
	img2paragraph_path = '../data/img2paragraph_modify_batch'
	pretrained_embed_matrix_path = "../data/pretrained_embed_matrix.npy"
	caption_labels_path = "../data/caption_labels_350.pkl"
	# caption_labels_path = "../data/caption_labels_VB_JJ_NN_gzip.hkl"

	densecap_train_path = "../data/image_path/imgs_train_path.txt"
	densecap_validate_path = "../data/image_path/imgs_val_path.txt"
	densecap_test_path = "../data/image_path/imgs_test_path.txt"

	semi_dense_feats_files = ["../data/densecap/coco_train_output_0to2.hkl"]
							  # "../data/densecap/coco_train_output_2to4.hkl",
							  # "../data/densecap/coco_train_output_4to6.hkl",
							  # "../data/densecap/coco_train_output_6to8.hkl",
							  # "../data/densecap/coco_train_output_after8.hkl"]
 
	semi_captions_files = ["../data/annotations/coco_captions_0to2.npy"]
						   # "../data/annotations/coco_captions_2to4.npy",
						   # "../data/annotations/coco_captions_4to6.npy",
						   # "../data/annotations/coco_captions_6to8.npy",
						   # "../data/annotations/coco_captions_after8.npy"]


	# for test
	# semi_dense_feats_files = ["../data/densecap/coco_train_output_test.hkl",
	# 						  "../data/densecap/coco_train_output_0to2.hkl",
	# 						  "../data/densecap/coco_train_output_2to4.hkl",
	# 						  "../data/densecap/coco_train_output_4to6.hkl",
	# 						  "../data/densecap/coco_train_output_6to8.hkl",
	# 						  "../data/densecap/coco_train_output_after8.hkl"]
 
	# semi_captions_files = ["../data/annotations/coco_captions_test.npy",
	# 					   "../data/annotations/coco_captions_0to2.npy",
	# 					   "../data/annotations/coco_captions_2to4.npy",
	# 					   "../data/annotations/coco_captions_4to6.npy",
	# 					   "../data/annotations/coco_captions_6to8.npy",
	# 					   "../data/annotations/coco_captions_after8.npy"]
